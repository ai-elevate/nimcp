//=============================================================================
// nimcp_brain_init_structural.c - Structural Subsystems
//=============================================================================
/**
 * @file nimcp_brain_init_structural.c
 * @brief Structural Subsystems
 *
 * WHAT: Initialization functions for structural subsystems
 * WHY:  SRP refactoring - separate structural initialization logic
 * HOW:  Each function initializes a specific brain subsystem
 *
 * EXTRACTED FROM: nimcp_brain_init_subsystems.c
 * DATE: 2025-12-08
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_structural.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/error/nimcp_error_codes.h"

#define LOG_MODULE "BRAIN_INIT_STRUCTURAL"

// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Subsystem includes (add as needed based on functions)
#include "glial/integration/nimcp_glial_integration.h"
#include "glial/myelin_sheath/nimcp_myelin_sheath.h"
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/introspection/nimcp_connectivity_health.h"
#include "middleware/integration/nimcp_middleware_controller.h"
#include "core/axon/nimcp_axon.h"
#include "core/dendrite/nimcp_dendrite.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/salience/nimcp_salience.h"
#include "cognitive/consolidation/nimcp_consolidation.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "cognitive/epistemic/nimcp_epistemic_filter.h"
#include "cognitive/wellbeing/nimcp_wellbeing.h"
#include "plasticity/neuromodulators/nimcp_neuromod_pink_noise.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_spatial_neuromod.h"
#include "plasticity/attention/nimcp_attention.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include "cognitive/nimcp_fractal_cognitive.h"
#include "core/integration/nimcp_multimodal_integration.h"
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_audio_cortex.h"
#include "perception/nimcp_speech_cortex.h"
#include "nlp/nimcp_nlp.h"
#include "core/brain_regions/nimcp_brain_regions.h"
#include "core/events/nimcp_event_bus.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_emotional_tagging.h"
#include "cognitive/nimcp_emotional_system.h"
#include "cognitive/nimcp_executive.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/memory/nimcp_engram.h"
#include "cognitive/memory/nimcp_systems_consolidation.h"
#include "cognitive/memory/nimcp_wm_transfer.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
#include "cognitive/nimcp_mental_health.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_explanations.h"
#include "cognitive/nimcp_meta_learning.h"
#include "cognitive/nimcp_predictive.h"
#include "cognitive/nimcp_mirror_neurons.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "cognitive/global_workspace/nimcp_global_workspace_shannon.h"
#include "cognitive/nimcp_autobiographical_memory.h"
#include "middleware/training/nimcp_brain_training_integration.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "plasticity/homeostatic/nimcp_homeostatic.h"
#include "plasticity/dendritic/nimcp_dendritic.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "core/cortical_columns/nimcp_cortical_column.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>

//=============================================================================
// Structural Subsystems Implementation
//=============================================================================


/**
 * @brief Initialize Axon Network subsystem (Phase 1.5.6)
 *
 * WHAT: Create axon network for realistic signal propagation with conduction delays
 * WHY:  Enable biologically realistic action potential propagation between neurons
 * HOW:  Create axon for each neuron, configure myelination based on neuron type
 *
 * BIOLOGICAL INSPIRATION:
 * - Axonal conduction delays vary 0.5-100 m/s based on myelination (Waxman, 1977)
 * - Myelinated axons have saltatory conduction (Huxley & Stämpfli, 1949)
 * - Axon diameter affects conduction velocity (Hursh, 1939)
 *
 * PERFORMANCE TARGET: O(n) where n = num_neurons
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_axon_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if axons are disabled via config
    if (!brain->config.enable_axons) {
        brain->axon_network = NULL;
        return true;  // Not an error - axons disabled by config
    }

    // Check if lazy initialization is requested
    if (brain->config.lazy_axon_init) {
        brain->axon_network = NULL;
        // Axons will be created on first access via brain_ensure_axons()
        return true;
    }

    // Guard: No network
    if (!brain->network) {
        brain->axon_network = NULL;
        return true;  // Not an error - just no network yet
    }

    // Get base network for neuron access
    neural_network_t nn = adaptive_network_get_base_network(brain->network);
    if (!nn) {
        brain->axon_network = NULL;
        return true;  // Not an error - empty network
    }

    // Get neuron count using accessor function (opaque type pattern)
    uint32_t num_neurons = neural_network_get_num_neurons(nn);
    if (num_neurons == 0) {
        brain->axon_network = NULL;
        return true;  // Not an error - empty network
    }

    // Create axon network with capacity for all neurons
    axon_network_t* axon_net = axon_network_create(num_neurons);
    if (!axon_net) {
        // Axon network creation failed - continue without axons (graceful degradation)
        brain->axon_network = NULL;
        return true;  // Not fatal - direct connections still work
    }

    // Create axon for each neuron
    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t* neuron = neural_network_get_neuron(nn, i);
        if (!neuron) {
            continue;  // Skip invalid neurons
        }

        // Determine axon type based on neuron type
        axon_type_t axon_type = AXON_TYPE_UNMYELINATED;  // Default
        float myelination = 0.0F;

        // Excitatory neurons tend to have larger, myelinated axons
        if (neuron->type == NEURON_EXCITATORY) {
            axon_type = AXON_TYPE_MYELINATED;
            myelination = 0.6F + (float)(rand() % 40) / 100.0F;  // 0.6-1.0
        } else {
            // Inhibitory interneurons often have unmyelinated or lightly myelinated axons
            axon_type = AXON_TYPE_UNMYELINATED;
            myelination = (float)(rand() % 30) / 100.0F;  // 0.0-0.3
        }

        // Calculate axon properties
        float length = 100.0F + (float)(rand() % 400);  // 100-500 μm
        float diameter = 0.5F + (float)(rand() % 20) / 10.0F;  // 0.5-2.5 μm

        // Create axon using the axon_create API
        // Parameters: id, type, source_neuron_id, target_synapse_id, length, diameter
        axon_t* axon = axon_create(i, axon_type, neuron->id, 0, length, diameter);
        if (axon) {
            // Set myelination level
            axon_set_myelination(axon, myelination);

            // Add to network
            if (axon_network_add(axon_net, axon)) {
                neuron->axon_id = axon->id;
            } else {
                axon_destroy(axon);
                neuron->axon_id = 0;
            }
        } else {
            neuron->axon_id = 0;  // No axon - direct connection
        }
    }

    brain->axon_network = axon_net;
    return true;
}


/**
 * @brief Initialize Dendrite Network subsystem (Phase 1.5.7)
 *
 * WHAT: Create dendrite network for spatiotemporal signal integration
 * WHY:  Enable biologically realistic dendritic computation and synaptic integration
 * HOW:  Create dendrites for each neuron, add segments and spines for synaptic inputs
 *
 * BIOLOGICAL INSPIRATION:
 * - Dendritic integration uses cable theory (Rall, 1959)
 * - Spines are primary sites of excitatory input (Yuste, 2010)
 * - Dendrites perform nonlinear computation (Polsky et al., 2004)
 *
 * PERFORMANCE TARGET: O(n) where n = num_neurons
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_dendrite_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if dendrites are disabled via config
    if (!brain->config.enable_dendrites) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - dendrites disabled by config
    }

    // Check if lazy initialization is requested
    if (brain->config.lazy_dendrite_init) {
        brain->dendrite_network = NULL;
        // Dendrites will be created on first access via brain_ensure_dendrites()
        return true;
    }

    // Guard: No network
    if (!brain->network) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - just no network yet
    }

    // Get base network for neuron access
    neural_network_t nn = adaptive_network_get_base_network(brain->network);
    if (!nn) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - empty network
    }

    // Get neuron count using accessor function (opaque type pattern)
    uint32_t num_neurons = neural_network_get_num_neurons(nn);
    if (num_neurons == 0) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - empty network
    }

    // Create dendrite network with capacity for all neurons (each can have multiple dendrites)
    // Estimate: ~2 dendrites per neuron on average (basal + apical for pyramidal)
    dendrite_network_t* dend_net = dendrite_network_create(num_neurons * 2);
    if (!dend_net) {
        // Dendrite network creation failed - continue without dendrites (graceful degradation)
        brain->dendrite_network = NULL;
        return true;  // Not fatal - direct inputs still work
    }

    // Create dendrites for each neuron
    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t* neuron = neural_network_get_neuron(nn, i);
        if (!neuron) {
            continue;  // Skip invalid neurons
        }

        // Initialize dendrite tracking for this neuron
        // Allocate space for up to 4 dendrites per neuron
        neuron->dendrite_ids = (uint32_t*)nimcp_calloc(4, sizeof(uint32_t));
        neuron->num_dendrites = 0;

        if (!neuron->dendrite_ids) {
            continue;  // Skip this neuron on allocation failure
        }

        // Determine dendrite configuration based on neuron type
        uint32_t num_dendrites_to_create = 1;  // Default: 1 dendrite
        dendrite_type_t primary_type = DENDRITE_TYPE_BASAL;

        // Excitatory neurons (pyramidal-like) have more elaborate dendritic trees
        if (neuron->type == NEURON_EXCITATORY) {
            num_dendrites_to_create = 2;  // Basal + apical
            primary_type = DENDRITE_TYPE_BASAL;
        }

        // Create dendrites for this neuron
        for (uint32_t d = 0; d < num_dendrites_to_create && neuron->num_dendrites < 4; d++) {
            dendrite_type_t dtype = (d == 0) ? primary_type : DENDRITE_TYPE_APICAL;

            dendrite_config_t dend_config = {
                .id = dend_net->num_dendrites,  // Sequential ID
                .type = dtype,
                .target_neuron_id = neuron->id,
                .total_length = 150.0F + (float)(rand() % 200),  // 150-350 μm
                .mean_diameter = 1.0F + (float)(rand() % 20) / 10.0F,  // 1.0-3.0 μm
                .start_pos = {0.0F, 0.0F, 0.0F},
                .integration_window_ms = 15.0F + (float)(rand() % 20),  // 15-35 ms
                .structural_plasticity = 0.01F,
                .ltp_threshold = 0.7F + (float)(rand() % 20) / 100.0F,  // 0.7-0.9
                .ltd_threshold = 0.2F + (float)(rand() % 20) / 100.0F   // 0.2-0.4
            };

            // Create dendrite
            dendrite_t* dendrite = dendrite_create(&dend_config);
            if (!dendrite) {
                continue;  // Skip on creation failure
            }

            // Create initial segments (3-5 compartments)
            uint32_t num_segments = 3 + (rand() % 3);  // 3-5 segments
            segment_config_t* seg_configs = (segment_config_t*)nimcp_calloc(
                num_segments, sizeof(segment_config_t));

            if (seg_configs) {
                float path_dist = 0.0F;
                for (uint32_t s = 0; s < num_segments; s++) {
                    seg_configs[s].type = (s == 0) ? DENDRITE_SEGMENT_PROXIMAL :
                                          (s == num_segments - 1) ? DENDRITE_SEGMENT_TERMINAL :
                                          DENDRITE_SEGMENT_SHAFT;
                    seg_configs[s].parent_segment = (s == 0) ? UINT32_MAX : (s - 1);
                    seg_configs[s].length = 30.0F + (float)(rand() % 40);  // 30-70 μm
                    seg_configs[s].diameter = dend_config.mean_diameter *
                                             (1.0F - 0.1F * s);  // Taper
                    seg_configs[s].path_distance = path_dist;
                    path_dist += seg_configs[s].length;
                    seg_configs[s].has_active_properties = (dtype == DENDRITE_TYPE_APICAL);
                }

                dendrite_create_segments(dendrite, num_segments, seg_configs);
                nimcp_free(seg_configs);
            }

            // Add dendrite to network
            if (dendrite_network_add(dend_net, dendrite)) {
                neuron->dendrite_ids[neuron->num_dendrites] = dendrite->id;
                neuron->num_dendrites++;
            } else {
                dendrite_destroy(dendrite);
            }
        }
    }

    brain->dendrite_network = dend_net;
    return true;
}


//=============================================================================
// Phase CC-1: Cortical Columns Subsystem Initialization (Tier 0.65)
//=============================================================================

/**
 * @brief Initialize cortical columns subsystem
 *
 * WHAT: Creates hierarchical cortical column architecture with minicolumns,
 *       hypercolumns, laminar organization, and topographic maps.
 *
 * WHY:  Implements biologically-realistic columnar organization (Douglas & Martin 1991)
 *       for feature detection and competitive dynamics in neural processing.
 *
 * HOW:  1. Create cortical column memory pool for efficient allocation
 *       2. Create hypercolumns based on brain region requirements
 *       3. Initialize 6-layer laminar structure for each region
 *       4. Set up canonical microcircuit connectivity
 *       5. Create topographic maps for sensory processing
 *       6. Initialize orientation columns for V1 (if visual enabled)
 *       7. Create feature hypercolumns for multi-dimensional features
 *
 * ARCHITECTURE:
 *   Hypercolumn (100K neurons):
 *   - Contains ~100 minicolumns
 *   - Each minicolumn ~80-100 neurons
 *   - Competition modes: winner-take-all, k-winners, softmax
 *   - Lateral inhibition (Mexican hat) for sharpening
 *
 * @param brain Brain to initialize cortical columns for
 * @return true on success (or if disabled), false on fatal error
 *
 * COMPLEXITY: O(H × M × N) where H=hypercolumns, M=minicolumns, N=neurons
 * THREAD-SAFE: Yes (pool is thread-safe after creation)
 *
 * @version 2.7.0 Phase CC-1
 * @date 2025-11-25
 */
bool nimcp_brain_factory_init_cortical_columns_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        set_error("nimcp_brain_factory_init_cortical_columns_subsystem: NULL brain");
        return false;
    }

    // Check if cortical columns are enabled
    if (!brain->config.enable_cortical_columns) {
        // Initialize pointers to NULL for safe cleanup
        brain->cortical_column_pool = NULL;
        brain->hypercolumns = NULL;
        brain->num_hypercolumns = 0;
        brain->laminar_system = NULL;
        brain->columnar_connectivity = NULL;
        brain->visual_topographic_map = NULL;
        brain->auditory_topographic_map = NULL;
        brain->somatosensory_topographic_map = NULL;
        brain->orientation_hypercolumns = NULL;
        brain->num_orientation_hypercolumns = 0;
        brain->feature_hypercolumns = NULL;
        brain->num_feature_hypercolumns = 0;
        brain->enable_cortical_columns = false;
        brain->last_cortical_update_us = 0;
        return true;  // Disabled, not an error
    }

    // Check for lazy initialization mode - defer allocation until first use
    if (brain->config.lazy_cortical_init) {
        // Mark as needing initialization but don't allocate yet
        brain->cortical_column_pool = NULL;
        brain->hypercolumns = NULL;
        brain->num_hypercolumns = 0;
        brain->laminar_system = NULL;
        brain->columnar_connectivity = NULL;
        brain->visual_topographic_map = NULL;
        brain->auditory_topographic_map = NULL;
        brain->somatosensory_topographic_map = NULL;
        brain->orientation_hypercolumns = NULL;
        brain->num_orientation_hypercolumns = 0;
        brain->feature_hypercolumns = NULL;
        brain->num_feature_hypercolumns = 0;
        brain->enable_cortical_columns = true;  // Mark as enabled but lazy
        brain->cortical_needs_lazy_init = true;  // Flag for lazy initialization
        brain->last_cortical_update_us = 0;
        LOG_INFO("Cortical columns subsystem deferred (lazy initialization)");
        return true;
    }

    // Check if already initialized (prevent double initialization)
    if (brain->cortical_column_pool) {
        return true;  // Already initialized
    }

    LOG_INFO("Initializing cortical columns subsystem (Phase CC-1)");

    // Get configuration parameters with defaults
    uint32_t num_hypercolumns = brain->config.num_hypercolumns > 0 ?
        brain->config.num_hypercolumns : 10;  // Default: 10 hypercolumns
    uint32_t minicolumns_per_hc = brain->config.minicolumns_per_hypercolumn > 0 ?
        brain->config.minicolumns_per_hypercolumn : 100;  // Default: 100 minicolumns
    uint32_t neurons_per_mc = brain->config.neurons_per_minicolumn > 0 ?
        brain->config.neurons_per_minicolumn : 80;  // Default: 80 neurons

    // Step 1: Create cortical column memory pool
    cortical_column_pool_config_t pool_config = {
        .max_minicolumns = num_hypercolumns * minicolumns_per_hc,
        .max_hypercolumns = num_hypercolumns,
        .max_neurons_per_minicolumn = neurons_per_mc,
        .enable_cow_support = brain->is_cow_clone || brain->can_use_readonly
    };

    brain->cortical_column_pool = cortical_column_pool_create(&pool_config);
    if (!brain->cortical_column_pool) {
        LOG_WARNING("Failed to create cortical column pool - continuing without columnar organization");
        brain->enable_cortical_columns = false;
        return true;  // Non-fatal: brain still works without columns
    }

    // Step 2: Create hypercolumns array
    brain->hypercolumns = (hypercolumn_t**)nimcp_calloc(num_hypercolumns, sizeof(hypercolumn_t*));
    if (!brain->hypercolumns) {
        LOG_WARNING("Failed to allocate hypercolumns array");
        cortical_column_pool_destroy(brain->cortical_column_pool);
        brain->cortical_column_pool = NULL;
        brain->enable_cortical_columns = false;
        return true;  // Non-fatal
    }

    brain->num_hypercolumns = num_hypercolumns;

    // Step 3: Initialize laminar structure if enabled
    if (brain->config.enable_laminar_structure) {
        // Create default laminar structure for cortical region
        // Pass NULL to use default layer configurations
        brain->laminar_system = laminar_structure_create(NULL);
        if (!brain->laminar_system) {
            LOG_WARNING("Failed to create laminar structure - continuing without layers");
        } else {
            LOG_INFO("Created 6-layer laminar structure with default configuration");
        }
    }

    // Step 4: Initialize columnar connectivity if enabled
    if (brain->config.enable_columnar_connectivity) {
        // Create canonical microcircuit connectivity
        // Estimate max connections: ~1000 per minicolumn * num_minicolumns
        uint32_t max_connections = num_hypercolumns * minicolumns_per_hc * 100;

        brain->columnar_connectivity = columnar_connectivity_create(max_connections);
        if (!brain->columnar_connectivity) {
            LOG_WARNING("Failed to create columnar connectivity - continuing without connectivity");
        } else {
            LOG_INFO("Created canonical microcircuit connectivity (max %u connections)", max_connections);
        }
    }

    // Step 5: Create topographic maps if sensory processing is enabled
    if (brain->config.enable_visual_topographic && brain->config.enable_visual_cortex) {
        // Create retinotopic map for V1 (log-polar mapping)
        retinotopic_params_t visual_params = {
            .foveal_radius = 2.0F,            // 2 degrees foveal region
            .cortical_magnification = 20.0F,  // 20 mm/degree at fovea
            .log_polar_a = 0.5F,              // Log-polar offset
            .aspect_ratio = 1.0F,             // Circular
            .eccentricity_half = 1.5F,        // E₂ for magnification falloff
            .angle_coverage = 2.0F * 3.14159F // Full visual field
        };
        // Default cortical dimensions: 64x64 columns
        uint32_t cortical_width = 64;
        uint32_t cortical_height = 64;

        brain->visual_topographic_map = topographic_map_create_retinotopic(
            &visual_params, cortical_width, cortical_height);
        if (!brain->visual_topographic_map) {
            LOG_WARNING("Failed to create visual topographic map");
        } else {
            LOG_INFO("Created retinotopic (log-polar) map for V1 (%ux%u)", cortical_width, cortical_height);
        }
    }

    if (brain->config.enable_auditory_topographic && brain->config.enable_audio_cortex) {
        // Create tonotopic map for A1 (logarithmic frequency mapping)
        tonotopic_params_t auditory_params = {
            .min_frequency = 20.0F,            // 20 Hz
            .max_frequency = 20000.0F,         // 20 kHz
            .octave_span = 1.0F,               // 1 octave per unit distance
            .is_logarithmic = true,            // Log frequency scale
            .q_factor = 10.0F                  // Constant Q bandwidth
        };
        // Default: 128 frequency bands (about 10 bands per octave over ~10 octaves)
        uint32_t num_frequency_bands = 128;

        brain->auditory_topographic_map = topographic_map_create_tonotopic(
            &auditory_params, num_frequency_bands);
        if (!brain->auditory_topographic_map) {
            LOG_WARNING("Failed to create auditory topographic map");
        } else {
            LOG_INFO("Created tonotopic (logarithmic) map for A1 (%u bands)", num_frequency_bands);
        }
    }

    if (brain->config.enable_somatosensory_topographic) {
        // Create somatotopic map for S1 (homunculus)
        // Default: 20 body regions (standard homunculus)
        uint32_t num_body_regions = 20;
        brain->somatosensory_topographic_map = topographic_map_create_somatotopic(num_body_regions);
        if (!brain->somatosensory_topographic_map) {
            LOG_WARNING("Failed to create somatosensory topographic map");
        } else {
            LOG_INFO("Created somatotopic (homunculus) map for S1 (%u regions)", num_body_regions);
        }
    }

    // Step 6: Initialize orientation columns if visual processing enabled
    if (brain->config.enable_orientation_columns && brain->config.enable_visual_cortex) {
        uint32_t num_orient_cols = brain->config.num_orientation_columns > 0 ?
            brain->config.num_orientation_columns : 16;  // Default: 16 orientations (0-180° in 11.25° steps)
        float spatial_frequency = 2.0F;  // Default: 2 cycles/degree
        float tuning_width = 30.0F;      // Default: 30° half-width at half-height

        // Create orientation hypercolumns (one per visual hypercolumn)
        uint32_t num_orient_hc = num_hypercolumns;  // Match hypercolumn count
        brain->orientation_hypercolumns = (orientation_hypercolumn_t**)nimcp_calloc(
            num_orient_hc, sizeof(orientation_hypercolumn_t*));

        if (brain->orientation_hypercolumns) {
            for (uint32_t i = 0; i < num_orient_hc; i++) {
                brain->orientation_hypercolumns[i] = orientation_hypercolumn_create(
                    num_orient_cols, spatial_frequency, tuning_width);
                if (!brain->orientation_hypercolumns[i]) {
                    LOG_WARNING("Failed to create orientation hypercolumn %u", i);
                }
            }
            brain->num_orientation_hypercolumns = num_orient_hc;
            LOG_INFO("Created %u orientation hypercolumns with %u columns each (sf=%.1f, tw=%.1f)",
                     num_orient_hc, num_orient_cols, spatial_frequency, tuning_width);
        }
    }

    // Step 7: Initialize feature hypercolumns if enabled
    if (brain->config.enable_feature_hypercolumns) {
        // Create feature hypercolumns for multi-dimensional feature spaces
        // Start with a basic orientation + spatial frequency hypercolumn
        uint32_t num_feat_hc = num_hypercolumns;  // One per spatial location

        brain->feature_hypercolumns = (feature_hypercolumn_t**)nimcp_calloc(
            num_feat_hc, sizeof(feature_hypercolumn_t*));

        if (brain->feature_hypercolumns) {
            for (uint32_t i = 0; i < num_feat_hc; i++) {
                // Create 2D feature space (orientation × spatial frequency)
                feature_dimension_t dims[2];
                dims[0] = feature_dimension_create(FEATURE_ORIENTATION, 0.0F, 180.0F, 16);
                feature_dimension_set_circular(&dims[0], true);
                dims[1] = feature_dimension_create(FEATURE_SPATIAL_FREQ, 0.5F, 8.0F, 8);

                brain->feature_hypercolumns[i] = feature_hypercolumn_create(dims, 2);
                if (!brain->feature_hypercolumns[i]) {
                    LOG_WARNING("Failed to create feature hypercolumn %u", i);
                }
            }
            brain->num_feature_hypercolumns = num_feat_hc;
            LOG_INFO("Created %u feature hypercolumns (orientation × spatial frequency)",
                     num_feat_hc);
        }
    }

    // Mark as enabled
    brain->enable_cortical_columns = true;
    brain->last_cortical_update_us = 0;

    LOG_INFO("Cortical columns subsystem initialized: %u hypercolumns, %s laminar, %s connectivity",
             brain->num_hypercolumns,
             brain->laminar_system ? "with" : "no",
             brain->columnar_connectivity ? "with" : "no");

    return true;
}
