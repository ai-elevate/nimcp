//=============================================================================
// nimcp_platform_tier.h - Platform Tier System (Portia Spider Foundation)
//=============================================================================
/**
 * @file nimcp_platform_tier.h
 * @brief Platform tier detection and configuration for constrained devices
 *
 * WHAT: Automatic platform classification and tier-specific configuration
 * WHY:  Enable NIMCP to run on diverse hardware (server → IoT/MCU)
 * HOW:  Detect system resources and apply tier-appropriate constraints
 *
 * PORTIA SPIDER FOUNDATION:
 * Named after Portia fimbriata, a jumping spider with extraordinary cognitive
 * abilities despite having only ~600,000 neurons and operating on minimal
 * energy. Just as Portia adapts its hunting strategies to available resources,
 * NIMCP adapts its architecture to platform constraints.
 *
 * DESIGN PHILOSOPHY:
 * Different platforms have vastly different resources:
 * - Desktop/Server: 8+ cores, 16GB+ RAM, GPU acceleration
 * - Laptop/Tablet: 4+ cores, 4GB+ RAM, modest compute
 * - Drone/Phone: 2-4 cores, 1-4GB RAM, power-constrained
 * - IoT/MCU: 1-2 cores, 64-256MB RAM, extreme constraints
 *
 * This system automatically detects platform capabilities and configures
 * NIMCP components (neuron counts, cortex resolution, cognitive modules)
 * to match available resources.
 *
 * TIER CLASSIFICATION:
 * - FULL: High-end workstations, servers (≥8 cores, ≥8GB RAM)
 * - MEDIUM: Laptops, tablets, dev boards (≥4 cores, ≥2GB RAM)
 * - CONSTRAINED: Phones, drones, edge devices (≥2 cores, ≥256MB RAM)
 * - MINIMAL: IoT, microcontrollers (≥1 core, ≥64MB RAM)
 *
 * COGNITIVE ADAPTATION:
 * Each tier enables/disables cognitive modules based on compute budget:
 * - FULL: All modules (curiosity, emotions, executive, meta-learning)
 * - MEDIUM: Core cognition (working memory, attention, emotions)
 * - CONSTRAINED: Essential modules (basic attention, simple memory)
 * - MINIMAL: Bare essentials (reactive processing only)
 *
 * INTEGRATION:
 * Brain creation → platform_tier_detect() → get tier config → apply limits
 * Visual cortex → tier config → set resolution/filters
 * Audio cortex → tier config → set sample rate/mel filters
 * Cognitive systems → tier config → enable/disable modules
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 2.8.0
 */

#ifndef NIMCP_PLATFORM_TIER_H
#define NIMCP_PLATFORM_TIER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "utils/platform/nimcp_system_resources.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Platform Tier Enumeration
//=============================================================================

/**
 * @brief Platform tier classification
 *
 * WHAT: Four-tier system from high-end to ultra-constrained
 * WHY:  Balance functionality vs resource constraints
 * HOW:  Detect RAM/cores and classify into appropriate tier
 */
typedef enum {
    /**
     * PLATFORM_TIER_FULL - High-end desktop/server
     * Hardware: ≥8 cores, ≥8GB RAM
     * Use cases: Research, development, training
     * Cognitive: All modules enabled
     */
    PLATFORM_TIER_FULL = 0,

    /**
     * PLATFORM_TIER_MEDIUM - Laptop/tablet/dev board
     * Hardware: ≥4 cores, ≥2GB RAM
     * Use cases: Development, inference, edge AI
     * Cognitive: Core modules (working memory, attention, emotions)
     */
    PLATFORM_TIER_MEDIUM = 1,

    /**
     * PLATFORM_TIER_CONSTRAINED - Phone/drone/embedded
     * Hardware: ≥2 cores, ≥256MB RAM
     * Use cases: Mobile robotics, drones, IoT gateways
     * Cognitive: Essential modules only (basic attention, simple memory)
     */
    PLATFORM_TIER_CONSTRAINED = 2,

    /**
     * PLATFORM_TIER_MINIMAL - IoT/MCU/ultra-constrained
     * Hardware: ≥1 core, ≥64MB RAM
     * Use cases: Sensors, actuators, minimal inference
     * Cognitive: Reactive processing only (no complex cognition)
     */
    PLATFORM_TIER_MINIMAL = 3,

    PLATFORM_TIER_COUNT = 4  /**< Number of tiers */
} platform_tier_t;

//=============================================================================
// Cognitive Module Flags (Bitmask)
//=============================================================================

/**
 * @brief Cognitive module enable flags
 *
 * WHAT: Bitmask for enabling/disabling cognitive modules
 * WHY:  Fine-grained control over which modules run at each tier
 * HOW:  Each bit represents a cognitive subsystem
 */
typedef enum {
    COGNITIVE_MODULE_NONE             = 0x00000000,  /**< No modules */

    // Core modules (highest priority)
    COGNITIVE_MODULE_ATTENTION        = 0x00000001,  /**< Attention system */
    COGNITIVE_MODULE_WORKING_MEMORY   = 0x00000002,  /**< Working memory */
    COGNITIVE_MODULE_SALIENCE         = 0x00000004,  /**< Salience detection */

    // Emotional modules
    COGNITIVE_MODULE_EMOTIONS         = 0x00000008,  /**< Emotional system */
    COGNITIVE_MODULE_EMOTIONAL_TAG    = 0x00000010,  /**< Emotional tagging */

    // Memory systems
    COGNITIVE_MODULE_SEMANTIC_MEMORY  = 0x00000020,  /**< Semantic memory */
    COGNITIVE_MODULE_EPISODIC_MEMORY  = 0x00000040,  /**< Episodic memory */
    COGNITIVE_MODULE_CONSOLIDATION    = 0x00000080,  /**< Memory consolidation */

    // Executive functions
    COGNITIVE_MODULE_EXECUTIVE        = 0x00000100,  /**< Executive control */
    COGNITIVE_MODULE_REASONING        = 0x00000200,  /**< Logical reasoning */
    COGNITIVE_MODULE_CURIOSITY        = 0x00000400,  /**< Curiosity system */

    // Meta-cognitive
    COGNITIVE_MODULE_META_LEARNING    = 0x00000800,  /**< Meta-learning */
    COGNITIVE_MODULE_INTROSPECTION    = 0x00001000,  /**< Introspection */
    COGNITIVE_MODULE_SELF_AWARENESS   = 0x00002000,  /**< Self-awareness */

    // Social cognition
    COGNITIVE_MODULE_THEORY_OF_MIND   = 0x00004000,  /**< Theory of mind */
    COGNITIVE_MODULE_MIRROR_NEURONS   = 0x00008000,  /**< Mirror neuron system */
    COGNITIVE_MODULE_EMPATHY          = 0x00010000,  /**< Empathetic response */

    // Advanced features
    COGNITIVE_MODULE_GLOBAL_WORKSPACE = 0x00020000,  /**< Global workspace */
    COGNITIVE_MODULE_PREDICTIVE       = 0x00040000,  /**< Predictive coding */
    COGNITIVE_MODULE_ETHICS           = 0x00080000,  /**< Ethical reasoning */

    // Perception (basic, usually enabled)
    COGNITIVE_MODULE_VISUAL_CORTEX    = 0x00100000,  /**< Visual processing */
    COGNITIVE_MODULE_AUDIO_CORTEX     = 0x00200000,  /**< Audio processing */

    // All modules
    COGNITIVE_MODULE_ALL              = 0xFFFFFFFF   /**< All modules */
} cognitive_module_flags_t;

//=============================================================================
// Visual Cortex Configuration
//=============================================================================

/**
 * @brief Visual cortex configuration per tier
 */
typedef struct {
    uint32_t max_input_width;      /**< Max input image width */
    uint32_t max_input_height;     /**< Max input image height */
    uint32_t num_filters_conv1;    /**< Conv layer 1 filters */
    uint32_t num_filters_conv2;    /**< Conv layer 2 filters */
    uint32_t max_feature_maps;     /**< Max feature maps */
    bool enable_pooling;           /**< Enable pooling layers */
    bool enable_attention;         /**< Enable visual attention */
} visual_cortex_tier_config_t;

//=============================================================================
// Audio Cortex Configuration
//=============================================================================

/**
 * @brief Audio cortex configuration per tier
 */
typedef struct {
    uint32_t max_sample_rate;      /**< Max sample rate (Hz) */
    uint32_t num_mel_filters;      /**< Mel-scale filter bank size */
    uint32_t num_mfcc;             /**< MFCC coefficients */
    uint32_t frame_size;           /**< Frame size (samples) */
    bool enable_attention;         /**< Enable auditory attention */
    bool enable_memory;            /**< Enable auditory memory */
} audio_cortex_tier_config_t;

//=============================================================================
// Platform Tier Configuration
//=============================================================================

/**
 * @brief Complete platform tier configuration
 *
 * WHAT: All configuration parameters for a specific tier
 * WHY:  Encapsulate tier-specific limits in one structure
 * HOW:  Pre-configured defaults for each tier, user can override
 *
 * THREAD SAFETY: Read-only after initialization (safe for concurrent reads)
 */
typedef struct {
    // Platform tier
    platform_tier_t tier;                    /**< Tier classification */

    // Core neural network constraints
    uint32_t max_neurons;                    /**< Max neurons in brain */
    uint32_t max_synapses_per_neuron;        /**< Max synapses per neuron */
    uint32_t initial_neurons;                /**< Initial neuron count */

    // Cognitive modules enabled
    uint32_t cognitive_modules_enabled;      /**< Bitmask of enabled modules */

    // Visual cortex configuration
    visual_cortex_tier_config_t visual;      /**< Visual cortex config */

    // Audio cortex configuration
    audio_cortex_tier_config_t audio;        /**< Audio cortex config */

    // Resource budgets
    uint32_t memory_budget_mb;               /**< Memory budget (MB) */
    uint64_t compute_budget_ops;             /**< Compute budget (ops/sec) */
    uint32_t max_threads;                    /**< Max parallel threads */

    // Feature flags
    bool enable_gpu;                         /**< Enable GPU acceleration */
    bool enable_bio_async;                   /**< Enable bio-async messaging */
    bool enable_plasticity;                  /**< Enable synaptic plasticity */
    bool enable_neuromodulation;             /**< Enable neuromodulators */
    bool enable_checkpointing;               /**< Enable state checkpointing */

    // Performance tuning
    uint32_t update_batch_size;              /**< Neurons per batch update */
    uint32_t spike_buffer_size;              /**< Spike event buffer size */
    float sampling_rate;                     /**< State sampling rate (0-1) */
} platform_tier_config_t;

//=============================================================================
// API Functions
//=============================================================================

/**
 * @brief Detect platform tier from system resources
 *
 * WHAT: Automatically classify platform into appropriate tier
 * WHY:  Enable automatic configuration without user intervention
 * HOW:  Query system resources (RAM, cores) and apply thresholds
 *
 * DETECTION ALGORITHM:
 * 1. Query system_resources_query() for RAM and CPU info
 * 2. Apply tier thresholds:
 *    - FULL: ≥8GB RAM AND ≥8 cores
 *    - MEDIUM: ≥2GB RAM AND ≥4 cores
 *    - CONSTRAINED: ≥256MB RAM AND ≥2 cores
 *    - MINIMAL: Otherwise
 * 3. Return detected tier
 *
 * THREAD SAFETY: Thread-safe (calls system_resources_query internally)
 *
 * @return Detected platform tier
 *
 * EXAMPLE:
 * @code
 * platform_tier_t tier = platform_tier_detect();
 * platform_tier_config_t config = platform_tier_get_config(tier);
 * brain_t* brain = brain_create(config.max_neurons, config.initial_neurons);
 * @endcode
 */
platform_tier_t platform_tier_detect(void);

/**
 * @brief Get pre-configured settings for specific tier
 *
 * WHAT: Retrieve default configuration for a given tier
 * WHY:  Provide sensible defaults tuned for each hardware class
 * HOW:  Return pre-computed configuration structure
 *
 * THREAD SAFETY: Thread-safe (returns const data)
 *
 * @param tier Platform tier
 * @return Configuration for that tier (safe to copy)
 *
 * EXAMPLE:
 * @code
 * platform_tier_config_t config = platform_tier_get_config(PLATFORM_TIER_MEDIUM);
 * printf("Max neurons: %u\n", config.max_neurons);
 * printf("Visual resolution: %ux%u\n", config.visual.max_input_width,
 *        config.visual.max_input_height);
 * @endcode
 */
platform_tier_config_t platform_tier_get_config(platform_tier_t tier);

/**
 * @brief Get human-readable tier name
 *
 * WHAT: Convert tier enum to descriptive string
 * WHY:  Useful for logging, debugging, user messages
 * HOW:  Simple lookup table
 *
 * THREAD SAFETY: Thread-safe (returns const string)
 *
 * @param tier Platform tier
 * @return Tier name string (e.g., "FULL", "MEDIUM", "CONSTRAINED", "MINIMAL")
 *
 * EXAMPLE:
 * @code
 * platform_tier_t tier = platform_tier_detect();
 * printf("Running on %s platform\n", platform_tier_get_name(tier));
 * // Output: "Running on MEDIUM platform"
 * @endcode
 */
const char* platform_tier_get_name(platform_tier_t tier);

/**
 * @brief Check if cognitive module is enabled at tier
 *
 * WHAT: Test if specific module is available at given tier
 * WHY:  Allow runtime checks before using optional modules
 * HOW:  Bitwise AND with tier's enabled module bitmask
 *
 * THREAD SAFETY: Thread-safe
 *
 * @param tier Platform tier
 * @param module Module flag to check (cognitive_module_flags_t)
 * @return true if module is enabled at this tier, false otherwise
 *
 * EXAMPLE:
 * @code
 * platform_tier_t tier = platform_tier_detect();
 * if (platform_tier_can_enable_module(tier, COGNITIVE_MODULE_CURIOSITY)) {
 *     // Safe to use curiosity system
 *     curiosity_system_init();
 * } else {
 *     printf("Curiosity disabled on this platform\n");
 * }
 * @endcode
 */
bool platform_tier_can_enable_module(platform_tier_t tier,
                                      cognitive_module_flags_t module);

/**
 * @brief Get recommended neuron count for tier
 *
 * WHAT: Calculate appropriate neuron count based on tier and available resources
 * WHY:  Balance functionality with memory constraints
 * HOW:  Use tier defaults, constrained by actual available RAM
 *
 * THREAD SAFETY: Thread-safe
 *
 * @param tier Platform tier
 * @param resources Current system resources (from system_resources_query)
 * @return Recommended neuron count
 *
 * EXAMPLE:
 * @code
 * system_resources_t resources;
 * system_resources_query(&resources);
 * platform_tier_t tier = platform_tier_detect();
 * uint32_t neurons = platform_tier_recommend_neuron_count(tier, &resources);
 * @endcode
 */
uint32_t platform_tier_recommend_neuron_count(platform_tier_t tier,
                                               const system_resources_t* resources);

/**
 * @brief Validate custom config against tier constraints
 *
 * WHAT: Check if user-provided config exceeds tier limits
 * WHY:  Prevent OOM/performance issues from over-ambitious configs
 * HOW:  Compare user config against tier maximums
 *
 * THREAD SAFETY: Thread-safe (read-only operations)
 *
 * @param tier Platform tier
 * @param config User configuration to validate
 * @param error_msg Buffer for error message (can be NULL)
 * @param error_msg_len Size of error message buffer
 * @return true if config is valid for tier, false if it exceeds limits
 *
 * EXAMPLE:
 * @code
 * platform_tier_config_t config = platform_tier_get_config(tier);
 * config.max_neurons = 10000000;  // User wants 10M neurons
 *
 * char error[256];
 * if (!platform_tier_validate_config(tier, &config, error, sizeof(error))) {
 *     printf("Invalid config: %s\n", error);
 * }
 * @endcode
 */
bool platform_tier_validate_config(platform_tier_t tier,
                                    const platform_tier_config_t* config,
                                    char* error_msg,
                                    size_t error_msg_len);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_PLATFORM_TIER_H
