//=============================================================================
// nimcp_platform_tier.c - Platform Tier System Implementation
//=============================================================================
/**
 * @file nimcp_platform_tier.c
 * @brief Platform tier detection and configuration implementation
 *
 * WHAT: Automatic platform classification and tier-specific configuration
 * WHY:  Enable NIMCP to run on diverse hardware (server → IoT/MCU)
 * HOW:  Detect system resources and apply tier-appropriate constraints
 *
 * IMPLEMENTATION NOTES:
 * - Tier configs are statically defined for fast access
 * - Detection uses system_resources_query() for accurate hardware info
 * - Conservative defaults to prevent OOM on constrained devices
 * - Modular design allows easy addition of new tiers
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include "utils/platform/nimcp_platform_tier.h"
#include "utils/platform/nimcp_system_resources.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"

#include <stdio.h>
#include <string.h>

//=============================================================================
// Tier Detection Thresholds
//=============================================================================

/**
 * WHAT: Hardware thresholds for tier classification
 * WHY:  Clear, conservative boundaries for automatic detection
 * HOW:  Based on typical hardware capabilities and NIMCP resource needs
 */

// FULL tier: High-end workstation/server
#define TIER_FULL_MIN_RAM_MB        8192    // 8GB RAM
#define TIER_FULL_MIN_CORES         8       // 8 cores

// MEDIUM tier: Laptop/tablet/dev board
#define TIER_MEDIUM_MIN_RAM_MB      2048    // 2GB RAM
#define TIER_MEDIUM_MIN_CORES       4       // 4 cores

// CONSTRAINED tier: Phone/drone/embedded
#define TIER_CONSTRAINED_MIN_RAM_MB 256     // 256MB RAM
#define TIER_CONSTRAINED_MIN_CORES  2       // 2 cores

// MINIMAL tier: IoT/MCU (everything below CONSTRAINED)
// No minimums - anything that doesn't meet CONSTRAINED goes here

//=============================================================================
// Pre-configured Tier Settings
//=============================================================================

/**
 * WHAT: Statically defined configurations for each tier
 * WHY:  Fast lookups, no dynamic allocation, cache-friendly
 * HOW:  Carefully tuned defaults based on hardware class
 *
 * TUNING PHILOSOPHY:
 * - Conservative memory budgets (leave room for OS, other processes)
 * - Progressive feature reduction (graceful degradation)
 * - Essential modules prioritized on constrained platforms
 * - Visual/audio configs scale with available compute
 */

static const platform_tier_config_t TIER_CONFIGS[PLATFORM_TIER_COUNT] = {
    // ========================================================================
    // PLATFORM_TIER_FULL - High-end desktop/server
    // ========================================================================
    {
        .tier = PLATFORM_TIER_FULL,

        // Core neural network - Large-scale brain
        .max_neurons = 1000000,              // 1M neurons (research-scale)
        .max_synapses_per_neuron = 10000,    // Dense connectivity
        .initial_neurons = 10000,            // Start with 10K neurons

        // Cognitive modules - ALL ENABLED
        .cognitive_modules_enabled =
            COGNITIVE_MODULE_ATTENTION |
            COGNITIVE_MODULE_WORKING_MEMORY |
            COGNITIVE_MODULE_SALIENCE |
            COGNITIVE_MODULE_EMOTIONS |
            COGNITIVE_MODULE_EMOTIONAL_TAG |
            COGNITIVE_MODULE_SEMANTIC_MEMORY |
            COGNITIVE_MODULE_EPISODIC_MEMORY |
            COGNITIVE_MODULE_CONSOLIDATION |
            COGNITIVE_MODULE_EXECUTIVE |
            COGNITIVE_MODULE_REASONING |
            COGNITIVE_MODULE_CURIOSITY |
            COGNITIVE_MODULE_META_LEARNING |
            COGNITIVE_MODULE_INTROSPECTION |
            COGNITIVE_MODULE_SELF_AWARENESS |
            COGNITIVE_MODULE_THEORY_OF_MIND |
            COGNITIVE_MODULE_MIRROR_NEURONS |
            COGNITIVE_MODULE_EMPATHY |
            COGNITIVE_MODULE_GLOBAL_WORKSPACE |
            COGNITIVE_MODULE_PREDICTIVE |
            COGNITIVE_MODULE_ETHICS |
            COGNITIVE_MODULE_VISUAL_CORTEX |
            COGNITIVE_MODULE_AUDIO_CORTEX,

        // Visual cortex - High resolution, full features
        .visual = {
            .max_input_width = 640,
            .max_input_height = 480,
            .num_filters_conv1 = 64,
            .num_filters_conv2 = 128,
            .max_feature_maps = 256,
            .enable_pooling = true,
            .enable_attention = true,
        },

        // Audio cortex - High fidelity
        .audio = {
            .max_sample_rate = 48000,        // CD quality
            .num_mel_filters = 128,
            .num_mfcc = 40,
            .frame_size = 2048,
            .enable_attention = true,
            .enable_memory = true,
        },

        // Resource budgets
        .memory_budget_mb = 4096,            // 4GB budget
        .compute_budget_ops = 10000000000ULL, // 10 GFLOPS
        .max_threads = 16,

        // Feature flags - All enabled
        .enable_gpu = true,
        .enable_bio_async = true,
        .enable_plasticity = true,
        .enable_neuromodulation = true,
        .enable_checkpointing = true,

        // Performance tuning
        .update_batch_size = 1000,
        .spike_buffer_size = 100000,
        .sampling_rate = 1.0f,               // Full sampling
    },

    // ========================================================================
    // PLATFORM_TIER_MEDIUM - Laptop/tablet/dev board
    // ========================================================================
    {
        .tier = PLATFORM_TIER_MEDIUM,

        // Core neural network - Medium-scale brain
        .max_neurons = 100000,               // 100K neurons
        .max_synapses_per_neuron = 1000,     // Moderate connectivity
        .initial_neurons = 5000,             // Start with 5K neurons

        // Cognitive modules - Core cognition enabled
        .cognitive_modules_enabled =
            COGNITIVE_MODULE_ATTENTION |
            COGNITIVE_MODULE_WORKING_MEMORY |
            COGNITIVE_MODULE_SALIENCE |
            COGNITIVE_MODULE_EMOTIONS |
            COGNITIVE_MODULE_EMOTIONAL_TAG |
            COGNITIVE_MODULE_SEMANTIC_MEMORY |
            COGNITIVE_MODULE_EPISODIC_MEMORY |
            COGNITIVE_MODULE_EXECUTIVE |
            COGNITIVE_MODULE_REASONING |
            COGNITIVE_MODULE_GLOBAL_WORKSPACE |
            COGNITIVE_MODULE_VISUAL_CORTEX |
            COGNITIVE_MODULE_AUDIO_CORTEX,

        // Visual cortex - Medium resolution
        .visual = {
            .max_input_width = 320,
            .max_input_height = 240,
            .num_filters_conv1 = 32,
            .num_filters_conv2 = 64,
            .max_feature_maps = 128,
            .enable_pooling = true,
            .enable_attention = true,
        },

        // Audio cortex - Moderate fidelity
        .audio = {
            .max_sample_rate = 22050,        // Half CD quality
            .num_mel_filters = 64,
            .num_mfcc = 20,
            .frame_size = 1024,
            .enable_attention = true,
            .enable_memory = true,
        },

        // Resource budgets
        .memory_budget_mb = 1024,            // 1GB budget
        .compute_budget_ops = 1000000000ULL, // 1 GFLOPS
        .max_threads = 8,

        // Feature flags - Most enabled
        .enable_gpu = false,                 // GPU optional
        .enable_bio_async = true,
        .enable_plasticity = true,
        .enable_neuromodulation = true,
        .enable_checkpointing = true,

        // Performance tuning
        .update_batch_size = 500,
        .spike_buffer_size = 50000,
        .sampling_rate = 0.8f,               // 80% sampling
    },

    // ========================================================================
    // PLATFORM_TIER_CONSTRAINED - Phone/drone/embedded
    // ========================================================================
    {
        .tier = PLATFORM_TIER_CONSTRAINED,

        // Core neural network - Small-scale brain
        .max_neurons = 10000,                // 10K neurons
        .max_synapses_per_neuron = 100,      // Sparse connectivity
        .initial_neurons = 1000,             // Start with 1K neurons

        // Cognitive modules - Essential only
        .cognitive_modules_enabled =
            COGNITIVE_MODULE_ATTENTION |
            COGNITIVE_MODULE_WORKING_MEMORY |
            COGNITIVE_MODULE_SALIENCE |
            COGNITIVE_MODULE_EMOTIONS |
            COGNITIVE_MODULE_VISUAL_CORTEX |
            COGNITIVE_MODULE_AUDIO_CORTEX,

        // Visual cortex - Low resolution
        .visual = {
            .max_input_width = 160,
            .max_input_height = 120,
            .num_filters_conv1 = 16,
            .num_filters_conv2 = 32,
            .max_feature_maps = 64,
            .enable_pooling = true,
            .enable_attention = false,       // Attention disabled
        },

        // Audio cortex - Low fidelity
        .audio = {
            .max_sample_rate = 16000,        // Phone quality
            .num_mel_filters = 32,
            .num_mfcc = 13,
            .frame_size = 512,
            .enable_attention = false,
            .enable_memory = false,          // No auditory memory
        },

        // Resource budgets
        .memory_budget_mb = 128,             // 128MB budget
        .compute_budget_ops = 100000000ULL,  // 100 MFLOPS
        .max_threads = 4,

        // Feature flags - Minimal features
        .enable_gpu = false,
        .enable_bio_async = true,            // Bio-async efficient
        .enable_plasticity = true,           // Basic learning
        .enable_neuromodulation = false,     // No neuromodulation
        .enable_checkpointing = false,       // No checkpoints

        // Performance tuning
        .update_batch_size = 100,
        .spike_buffer_size = 10000,
        .sampling_rate = 0.5f,               // 50% sampling
    },

    // ========================================================================
    // PLATFORM_TIER_MINIMAL - IoT/MCU/ultra-constrained
    // ========================================================================
    {
        .tier = PLATFORM_TIER_MINIMAL,

        // Core neural network - Tiny brain
        .max_neurons = 1000,                 // 1K neurons (Portia-scale!)
        .max_synapses_per_neuron = 10,       // Very sparse
        .initial_neurons = 100,              // Start with 100 neurons

        // Cognitive modules - Bare minimum
        .cognitive_modules_enabled =
            COGNITIVE_MODULE_ATTENTION |     // Basic attention only
            COGNITIVE_MODULE_VISUAL_CORTEX,  // Minimal vision

        // Visual cortex - Minimal resolution
        .visual = {
            .max_input_width = 64,
            .max_input_height = 48,
            .num_filters_conv1 = 8,
            .num_filters_conv2 = 16,
            .max_feature_maps = 32,
            .enable_pooling = false,         // No pooling
            .enable_attention = false,
        },

        // Audio cortex - Disabled (too expensive)
        .audio = {
            .max_sample_rate = 8000,         // Minimal
            .num_mel_filters = 16,
            .num_mfcc = 8,
            .frame_size = 256,
            .enable_attention = false,
            .enable_memory = false,
        },

        // Resource budgets
        .memory_budget_mb = 32,              // 32MB budget
        .compute_budget_ops = 10000000ULL,   // 10 MFLOPS
        .max_threads = 2,

        // Feature flags - Absolutely minimal
        .enable_gpu = false,
        .enable_bio_async = false,           // Too expensive
        .enable_plasticity = false,          // No learning
        .enable_neuromodulation = false,
        .enable_checkpointing = false,

        // Performance tuning
        .update_batch_size = 50,
        .spike_buffer_size = 1000,
        .sampling_rate = 0.25f,              // 25% sampling
    },
};

//=============================================================================
// Tier Name Strings
//=============================================================================

static const char* TIER_NAMES[PLATFORM_TIER_COUNT] = {
    "FULL",
    "MEDIUM",
    "CONSTRAINED",
    "MINIMAL"
};

//=============================================================================
// API Implementation
//=============================================================================

platform_tier_t platform_tier_detect(void)
{
    system_resources_t resources;

    // Query system resources
    if (!system_resources_query(&resources)) {
        // If query fails, assume minimal platform
        LOG_WARN("Failed to query system resources, defaulting to MINIMAL tier");
        return PLATFORM_TIER_MINIMAL;
    }

    // Apply tier detection logic
    // WHAT: Classify based on RAM and core count thresholds
    // WHY:  Both memory and compute affect NIMCP performance
    // HOW:  Use AND logic (both must meet threshold for tier)

    // Check FULL tier (≥8GB RAM AND ≥8 cores)
    if (resources.total_ram_mb >= TIER_FULL_MIN_RAM_MB &&
        resources.num_cpu_cores >= TIER_FULL_MIN_CORES) {
        LOG_INFO("Detected FULL platform tier: %llu MB RAM, %u cores",
                 (unsigned long long)resources.total_ram_mb,
                 resources.num_cpu_cores);
        return PLATFORM_TIER_FULL;
    }

    // Check MEDIUM tier (≥2GB RAM AND ≥4 cores)
    if (resources.total_ram_mb >= TIER_MEDIUM_MIN_RAM_MB &&
        resources.num_cpu_cores >= TIER_MEDIUM_MIN_CORES) {
        LOG_INFO("Detected MEDIUM platform tier: %llu MB RAM, %u cores",
                 (unsigned long long)resources.total_ram_mb,
                 resources.num_cpu_cores);
        return PLATFORM_TIER_MEDIUM;
    }

    // Check CONSTRAINED tier (≥256MB RAM AND ≥2 cores)
    if (resources.total_ram_mb >= TIER_CONSTRAINED_MIN_RAM_MB &&
        resources.num_cpu_cores >= TIER_CONSTRAINED_MIN_CORES) {
        LOG_INFO("Detected CONSTRAINED platform tier: %llu MB RAM, %u cores",
                 (unsigned long long)resources.total_ram_mb,
                 resources.num_cpu_cores);
        return PLATFORM_TIER_CONSTRAINED;
    }

    // Default to MINIMAL tier
    LOG_INFO("Detected MINIMAL platform tier: %llu MB RAM, %u cores",
             (unsigned long long)resources.total_ram_mb,
             resources.num_cpu_cores);
    return PLATFORM_TIER_MINIMAL;
}

platform_tier_config_t platform_tier_get_config(platform_tier_t tier)
{
    // Validate tier
    if (tier >= PLATFORM_TIER_COUNT) {
        LOG_WARN("Invalid tier %d, defaulting to MINIMAL", tier);
        tier = PLATFORM_TIER_MINIMAL;
    }

    // Return copy of config (safe for modification by caller)
    return TIER_CONFIGS[tier];
}

const char* platform_tier_get_name(platform_tier_t tier)
{
    // Validate tier
    if (tier >= PLATFORM_TIER_COUNT) {
        return "UNKNOWN";
    }

    return TIER_NAMES[tier];
}

bool platform_tier_can_enable_module(platform_tier_t tier,
                                      cognitive_module_flags_t module)
{
    // Validate tier
    if (tier >= PLATFORM_TIER_COUNT) {
        return false;
    }

    // Check if module bit is set in tier's enabled modules
    uint32_t enabled = TIER_CONFIGS[tier].cognitive_modules_enabled;
    return (enabled & module) != 0;
}

uint32_t platform_tier_recommend_neuron_count(platform_tier_t tier,
                                               const system_resources_t* resources)
{
    // Validate inputs
    if (tier >= PLATFORM_TIER_COUNT || !resources) {
        LOG_WARN("Invalid inputs to platform_tier_recommend_neuron_count");
        return 1000;  // Safe default
    }

    // Get tier defaults
    const platform_tier_config_t* config = &TIER_CONFIGS[tier];

    // Calculate max neurons based on available memory
    // WHAT: Use conservative estimate of memory per neuron
    // WHY:  Prevent OOM from over-optimistic sizing
    // HOW:  10KB per neuron for CPU mode, 80% safety margin

    const uint32_t KB_PER_NEURON = 10;
    const float SAFETY_MARGIN = 0.8f;

    uint64_t available_kb = resources->available_ram_mb * 1024;
    uint64_t safe_kb = (uint64_t)(available_kb * SAFETY_MARGIN);
    uint32_t max_by_memory = (uint32_t)(safe_kb / KB_PER_NEURON);

    // Take minimum of tier max and memory-based max
    uint32_t recommended = (max_by_memory < config->max_neurons) ?
                           max_by_memory : config->max_neurons;

    // Don't go below tier's initial neuron count
    if (recommended < config->initial_neurons) {
        recommended = config->initial_neurons;
    }

    LOG_DEBUG("Recommended neuron count for %s tier: %u "
              "(tier_max=%u, memory_max=%u, available_ram=%llu MB)",
              platform_tier_get_name(tier),
              recommended,
              config->max_neurons,
              max_by_memory,
              (unsigned long long)resources->available_ram_mb);

    return recommended;
}

bool platform_tier_validate_config(platform_tier_t tier,
                                    const platform_tier_config_t* config,
                                    char* error_msg,
                                    size_t error_msg_len)
{
    // Validate inputs
    if (tier >= PLATFORM_TIER_COUNT || !config) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len, "Invalid tier or NULL config");
        }
        return false;
    }

    // Get tier limits
    const platform_tier_config_t* limits = &TIER_CONFIGS[tier];

    // Check neuron count
    if (config->max_neurons > limits->max_neurons) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "max_neurons %u exceeds tier limit %u",
                     config->max_neurons, limits->max_neurons);
        }
        return false;
    }

    // Check synapses per neuron
    if (config->max_synapses_per_neuron > limits->max_synapses_per_neuron) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "max_synapses_per_neuron %u exceeds tier limit %u",
                     config->max_synapses_per_neuron,
                     limits->max_synapses_per_neuron);
        }
        return false;
    }

    // Check memory budget
    if (config->memory_budget_mb > limits->memory_budget_mb) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "memory_budget_mb %u exceeds tier limit %u",
                     config->memory_budget_mb, limits->memory_budget_mb);
        }
        return false;
    }

    // Check visual cortex resolution
    if (config->visual.max_input_width > limits->visual.max_input_width ||
        config->visual.max_input_height > limits->visual.max_input_height) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "Visual resolution %ux%u exceeds tier limit %ux%u",
                     config->visual.max_input_width,
                     config->visual.max_input_height,
                     limits->visual.max_input_width,
                     limits->visual.max_input_height);
        }
        return false;
    }

    // Check audio sample rate
    if (config->audio.max_sample_rate > limits->audio.max_sample_rate) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "Audio sample rate %u exceeds tier limit %u",
                     config->audio.max_sample_rate,
                     limits->audio.max_sample_rate);
        }
        return false;
    }

    // Check for enabled modules not supported at tier
    uint32_t unsupported = config->cognitive_modules_enabled &
                          ~limits->cognitive_modules_enabled;
    if (unsupported != 0) {
        if (error_msg && error_msg_len > 0) {
            snprintf(error_msg, error_msg_len,
                     "Config enables modules (0x%08X) not supported at %s tier",
                     unsupported, platform_tier_get_name(tier));
        }
        return false;
    }

    // Validation passed
    if (error_msg && error_msg_len > 0) {
        error_msg[0] = '\0';  // Clear error message
    }

    return true;
}

//=============================================================================
// Internal Utilities
//=============================================================================

/**
 * @brief Get module name for logging (internal helper)
 *
 * WHAT: Convert module flag to human-readable name
 * WHY:  Better logging and debugging
 * HOW:  Simple switch statement
 *
 * NOTE: Not exposed in public API (internal use only)
 */
static const char* get_module_name(cognitive_module_flags_t module)
{
    switch (module) {
        case COGNITIVE_MODULE_ATTENTION:        return "Attention";
        case COGNITIVE_MODULE_WORKING_MEMORY:   return "Working Memory";
        case COGNITIVE_MODULE_SALIENCE:         return "Salience";
        case COGNITIVE_MODULE_EMOTIONS:         return "Emotions";
        case COGNITIVE_MODULE_EMOTIONAL_TAG:    return "Emotional Tagging";
        case COGNITIVE_MODULE_SEMANTIC_MEMORY:  return "Semantic Memory";
        case COGNITIVE_MODULE_EPISODIC_MEMORY:  return "Episodic Memory";
        case COGNITIVE_MODULE_CONSOLIDATION:    return "Consolidation";
        case COGNITIVE_MODULE_EXECUTIVE:        return "Executive";
        case COGNITIVE_MODULE_REASONING:        return "Reasoning";
        case COGNITIVE_MODULE_CURIOSITY:        return "Curiosity";
        case COGNITIVE_MODULE_META_LEARNING:    return "Meta-Learning";
        case COGNITIVE_MODULE_INTROSPECTION:    return "Introspection";
        case COGNITIVE_MODULE_SELF_AWARENESS:   return "Self-Awareness";
        case COGNITIVE_MODULE_THEORY_OF_MIND:   return "Theory of Mind";
        case COGNITIVE_MODULE_MIRROR_NEURONS:   return "Mirror Neurons";
        case COGNITIVE_MODULE_EMPATHY:          return "Empathy";
        case COGNITIVE_MODULE_GLOBAL_WORKSPACE: return "Global Workspace";
        case COGNITIVE_MODULE_PREDICTIVE:       return "Predictive";
        case COGNITIVE_MODULE_ETHICS:           return "Ethics";
        case COGNITIVE_MODULE_VISUAL_CORTEX:    return "Visual Cortex";
        case COGNITIVE_MODULE_AUDIO_CORTEX:     return "Audio Cortex";
        default:                                return "Unknown";
    }
}

//=============================================================================
// Diagnostic Functions (Optional, for testing/debugging)
//=============================================================================

/**
 * @brief Print tier configuration summary
 *
 * WHAT: Output detailed config to console for debugging
 * WHY:  Helpful for verifying tier settings
 * HOW:  Format and print all config fields
 *
 * NOTE: This is a diagnostic function, not part of core API
 *       Can be conditionally compiled out in production builds
 */
void platform_tier_print_config(platform_tier_t tier)
{
    if (tier >= PLATFORM_TIER_COUNT) {
        printf("Invalid tier: %d\n", tier);
        return;
    }

    const platform_tier_config_t* cfg = &TIER_CONFIGS[tier];

    printf("\n========== Platform Tier: %s ==========\n", TIER_NAMES[tier]);
    printf("Core Neural Network:\n");
    printf("  Max neurons:              %u\n", cfg->max_neurons);
    printf("  Max synapses/neuron:      %u\n", cfg->max_synapses_per_neuron);
    printf("  Initial neurons:          %u\n", cfg->initial_neurons);

    printf("\nVisual Cortex:\n");
    printf("  Max resolution:           %ux%u\n",
           cfg->visual.max_input_width, cfg->visual.max_input_height);
    printf("  Conv1 filters:            %u\n", cfg->visual.num_filters_conv1);
    printf("  Conv2 filters:            %u\n", cfg->visual.num_filters_conv2);
    printf("  Pooling enabled:          %s\n", cfg->visual.enable_pooling ? "yes" : "no");
    printf("  Attention enabled:        %s\n", cfg->visual.enable_attention ? "yes" : "no");

    printf("\nAudio Cortex:\n");
    printf("  Max sample rate:          %u Hz\n", cfg->audio.max_sample_rate);
    printf("  Mel filters:              %u\n", cfg->audio.num_mel_filters);
    printf("  MFCC coefficients:        %u\n", cfg->audio.num_mfcc);
    printf("  Frame size:               %u\n", cfg->audio.frame_size);

    printf("\nResource Budgets:\n");
    printf("  Memory budget:            %u MB\n", cfg->memory_budget_mb);
    printf("  Compute budget:           %llu ops/sec\n",
           (unsigned long long)cfg->compute_budget_ops);
    printf("  Max threads:              %u\n", cfg->max_threads);

    printf("\nFeature Flags:\n");
    printf("  GPU enabled:              %s\n", cfg->enable_gpu ? "yes" : "no");
    printf("  Bio-async enabled:        %s\n", cfg->enable_bio_async ? "yes" : "no");
    printf("  Plasticity enabled:       %s\n", cfg->enable_plasticity ? "yes" : "no");
    printf("  Neuromodulation enabled:  %s\n", cfg->enable_neuromodulation ? "yes" : "no");
    printf("  Checkpointing enabled:    %s\n", cfg->enable_checkpointing ? "yes" : "no");

    printf("\nPerformance Tuning:\n");
    printf("  Update batch size:        %u\n", cfg->update_batch_size);
    printf("  Spike buffer size:        %u\n", cfg->spike_buffer_size);
    printf("  Sampling rate:            %.2f\n", cfg->sampling_rate);

    printf("\nCognitive Modules Enabled: 0x%08X\n", cfg->cognitive_modules_enabled);

    printf("=============================================\n\n");
}
