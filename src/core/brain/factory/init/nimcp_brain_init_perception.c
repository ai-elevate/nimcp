//=============================================================================
// nimcp_brain_init_perception.c - Perception and Sensory Subsystems
//=============================================================================
/**
 * @file nimcp_brain_init_perception.c
 * @brief Perception and Sensory Subsystems
 *
 * WHAT: Initialization functions for perception subsystems
 * WHY:  SRP refactoring - separate perception initialization logic
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

#include "core/brain/factory/init/nimcp_brain_init_perception.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/error/nimcp_error_codes.h"

#define LOG_MODULE "BRAIN_INIT_PERCEPTION"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_perception)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_perception_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_perception_mesh_registry = NULL;

nimcp_error_t brain_init_perception_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_perception_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_perception", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_perception";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_perception_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_perception_mesh_registry = registry;
    return err;
}

void brain_init_perception_mesh_unregister(void) {
    if (g_brain_init_perception_mesh_registry && g_brain_init_perception_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_perception_mesh_registry, g_brain_init_perception_mesh_id);
        g_brain_init_perception_mesh_id = 0;
        g_brain_init_perception_mesh_registry = NULL;
    }
}


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
// Perception and Sensory Subsystems Implementation
//=============================================================================


//=============================================================================
// TO COMPLETE THIS FILE:
// Run: sed -n '803,3821p' ../nimcp_brain_init.c >> nimcp_brain_init_subsystems.c
//
// This will extract lines 803-3821 from the original file which contains
// all 37 subsystem init functions.
//=============================================================================
bool nimcp_brain_factory_init_glial_subsystem(brain_t brain)
{
    if (!brain || !brain->network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "nimcp_brain_factory_init_glial_subsystem: invalid parameters");

            return false;
    }

    // Check if already initialized (prevent double initialization)
    if (brain->glial) {
        return true;  // Already initialized
    }

    // Check if glial integration is enabled
    if (!brain->config.enable_glial) {
        return true;  // Disabled, not an error
    }

    // Get base network for glial integration
    neural_network_t base = adaptive_network_get_base_network(brain->network);
    if (!base) {
        set_error("Failed to get base network for glial integration");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_glial_subsystem: base is NULL");
        return false;
    }

    // Create glial integration with reasonable max_mappings
    brain->glial = glial_integration_create(base, 1000);

    if (!brain->glial) {
        set_error("Failed to create glial integration");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_glial_subsystem: brain->glial is NULL");
        return false;
    }

    // Initialize myelin sheath network if enabled
    if (brain->config.enable_myelin_sheath) {
        // Check if already initialized
        if (!brain->myelin_sheath) {
            // Create myelin sheath network with default configuration
            // Capacity scales with oligodendrocyte count (each oligo can myelinate multiple axons)
            uint32_t myelin_capacity = brain->config.num_oligodendrocytes > 0
                ? brain->config.num_oligodendrocytes * 10
                : 1000;  // Default capacity
            brain->myelin_sheath = myelin_network_create_default(myelin_capacity);
            if (!brain->myelin_sheath) {
                set_error("Failed to create myelin sheath network");
                // Non-fatal: glial integration still works without myelin modeling
            } else {
                // Wire myelin sheath network to glial integration layer
                nimcp_result_t result = glial_integration_set_myelin_sheath_network(
                    brain->glial, brain->myelin_sheath);
                if (result != NIMCP_SUCCESS) {
                    set_error("Failed to wire myelin sheath to glial integration");
                    // Non-fatal: myelin network still usable standalone
                }
            }
        }
    }

    return true;
}

bool nimcp_brain_factory_init_multimodal_subsystems(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_multimodal_subsystems: brain is NULL");

            return false;
    }

    // Check if multi-modal processing is enabled
    if (!brain->config.enable_multimodal_integration) {
        // Even when multimodal is disabled, we still need integrated_feature_buffer
        // as a working buffer for direct-only predictions
        if (!brain->integrated_feature_buffer) {
            brain->integrated_feature_buffer = nimcp_calloc(brain->config.num_inputs, sizeof(float));
            if (!brain->integrated_feature_buffer) {
                set_error("Failed to allocate integrated feature buffer for direct predictions");
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_factory_init_multimodal_subsystems: brain->integrated_feature_buffer is NULL");
                return false;
            }
        }
        return true;
    }

    // If multimodal is enabled, check if already fully initialized
    // We check for visual/audio/speech cortices only if they should be enabled
    bool visual_needed = brain->config.enable_visual_cortex && brain->config.visual_feature_dim > 0;
    bool audio_needed = brain->config.enable_audio_cortex && brain->config.audio_feature_dim > 0;
    bool speech_needed = brain->config.enable_speech_cortex && brain->config.speech_feature_dim > 0;

    bool visual_ready = !visual_needed || brain->visual_cortex;
    bool audio_ready = !audio_needed || brain->audio_cortex;
    bool speech_ready = !speech_needed || brain->speech_cortex;

    // If all needed components are ready and multimodal layer exists, we're done
    if (brain->multimodal && visual_ready && audio_ready && speech_ready) {
        return true;  // Already fully initialized
    }

    // Initialize visual cortex (if enabled)
    if (brain->config.enable_visual_cortex && brain->config.visual_feature_dim > 0) {
        visual_cortex_config_t visual_config = {
            .input_width = 640,        // Default camera resolution
            .input_height = 480,
            .num_v1_filters = 32,      // 32 orientation-selective filters
            .feature_dim = brain->config.visual_feature_dim,
            .enable_attention = true,
            .enable_memory = true,

            // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
            .enable_fractal_topology = brain->config.enable_fractal_topology,
            .hub_ratio = 0.15F,        // 15% hub neurons (biological cortex ratio)
            .power_law_gamma = -2.1F,  // Cortical power-law exponent
            .internal_neurons = 32 * 10 // 10 neurons per filter (V1 columnar structure)
        };

        brain->visual_cortex = visual_cortex_create(&visual_config);
        if (!brain->visual_cortex) {
            set_error("Failed to create visual cortex");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_multimodal_subsystems: brain->visual_cortex is NULL");
            return false;
        }

        // Allocate visual feature buffer
        brain->visual_feature_buffer = nimcp_calloc(brain->config.visual_feature_dim, sizeof(float));
        if (!brain->visual_feature_buffer) {
            set_error("Failed to allocate visual feature buffer");
            visual_cortex_destroy(brain->visual_cortex);
            brain->visual_cortex = NULL;
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_factory_init_multimodal_subsystems: brain->visual_feature_buffer is NULL");
            return false;
        }
    }

    // Initialize audio cortex (if enabled)
    if (brain->config.enable_audio_cortex && brain->config.audio_feature_dim > 0) {
        audio_cortex_config_t audio_config = {
            .sample_rate = 16000,      // Default 16kHz audio
            .frame_size = 512,         // 32ms frames at 16kHz
            .num_freq_bins = 256,
            .num_mel_filters = 40,     // Standard for speech
            .num_mfcc = brain->config.audio_feature_dim,
            .num_channels = 1,         // Mono by default
            .feature_dim = brain->config.audio_feature_dim,
            .enable_attention = true,
            .enable_memory = true,

            // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
            .enable_fractal_topology = brain->config.enable_fractal_topology,
            .hub_ratio = 0.15F,        // 15% hub neurons (biological A1 ratio)
            .power_law_gamma = -2.1F,  // Tonotopic power-law exponent
            .internal_neurons = 40 * 10 // 10 neurons per mel filter (A1 tonotopic structure)
        };

        brain->audio_cortex = audio_cortex_create(&audio_config);
        if (!brain->audio_cortex) {
            set_error("Failed to create audio cortex");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_multimodal_subsystems: brain->audio_cortex is NULL");
            return false;
        }

        // Allocate audio feature buffer
        brain->audio_feature_buffer = nimcp_calloc(brain->config.audio_feature_dim, sizeof(float));
        if (!brain->audio_feature_buffer) {
            set_error("Failed to allocate audio feature buffer");
            audio_cortex_destroy(brain->audio_cortex);
            brain->audio_cortex = NULL;
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_factory_init_multimodal_subsystems: brain->audio_feature_buffer is NULL");
            return false;
        }
    }

    // Initialize speech cortex (Phase 8.8)
    if (brain->config.enable_speech_cortex && brain->config.speech_feature_dim > 0) {
        speech_cortex_config_t speech_config = speech_cortex_default_config();

        // Override defaults with brain config
        speech_config.sample_rate = 16000;        // Standard speech rate
        speech_config.frame_size_ms = 20;         // 20ms frames for phoneme analysis
        speech_config.num_phonemes = SPEECH_NUM_PHONEMES; // 44 phonemes (English)
        speech_config.feature_dim = brain->config.speech_feature_dim;
        speech_config.enable_wernicke = true;     // Enable word recognition
        speech_config.enable_prosody = true;      // Enable pitch/stress analysis
        speech_config.enable_memory = true;       // Enable phonological working memory

        // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
        speech_config.enable_fractal_topology = brain->config.enable_fractal_topology;
        speech_config.hub_ratio = 0.15F;          // 15% hub neurons (biological STG ratio)
        speech_config.power_law_gamma = -2.1F;    // Speech network power-law exponent
        speech_config.internal_neurons = SPEECH_NUM_PHONEMES * 10; // 10 neurons per phoneme

        brain->speech_cortex = speech_cortex_create(&speech_config);
        if (!brain->speech_cortex) {
            set_error("Failed to create speech cortex");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_multimodal_subsystems: brain->speech_cortex is NULL");
            return false;
        }

        // Allocate speech feature buffer
        brain->speech_feature_buffer = nimcp_calloc(brain->config.speech_feature_dim, sizeof(float));
        if (!brain->speech_feature_buffer) {
            set_error("Failed to allocate speech feature buffer");
            speech_cortex_destroy(brain->speech_cortex);
            brain->speech_cortex = NULL;
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_factory_init_multimodal_subsystems: brain->speech_feature_buffer is NULL");
            return false;
        }
    }

    // Initialize multi-modal integration layer
    uint32_t visual_dim = brain->config.enable_visual_cortex ? brain->config.visual_feature_dim : 0;
    uint32_t audio_dim = brain->config.enable_audio_cortex ? brain->config.audio_feature_dim : 0;
    uint32_t speech_dim = brain->config.enable_speech_cortex ? brain->config.speech_feature_dim : 0;
    // Direct dimension: Remaining space after visual, audio, and speech features
    uint32_t direct_dim = 0;
    if (brain->config.num_inputs > (visual_dim + audio_dim + speech_dim)) {
        direct_dim = brain->config.num_inputs - visual_dim - audio_dim - speech_dim;
    }

    if (visual_dim > 0 || audio_dim > 0 || speech_dim > 0 || direct_dim > 0) {
        // Phase 8.8: Speech is now a dedicated modality
        multimodal_config_t mm_config = multimodal_default_config(visual_dim, audio_dim, speech_dim, direct_dim);

        // Output dimension should match network input size
        mm_config.output_dim = brain->config.num_inputs;

        brain->multimodal = multimodal_integration_create(&mm_config);
        if (!brain->multimodal) {
            set_error("Failed to create multimodal integration layer");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_multimodal_subsystems: brain->multimodal is NULL");
            return false;
        }

        // Allocate integrated feature buffer
        brain->integrated_feature_buffer = nimcp_calloc(mm_config.output_dim, sizeof(float));
        if (!brain->integrated_feature_buffer) {
            set_error("Failed to allocate integrated feature buffer");
            multimodal_integration_destroy(brain->multimodal);
            brain->multimodal = NULL;
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_factory_init_multimodal_subsystems: brain->integrated_feature_buffer is NULL");
            return false;
        }
    }

    // Initialize NLP network (if multimodal or speech is enabled)
    if (brain->config.enable_multimodal_integration || brain->config.enable_speech_cortex) {

        // Configure NLP network with minimal config
        nlp_network_config_t nlp_config = {0};

        // NLP-specific parameters
        nlp_config.vocab_size = 10000;            // 10k token vocabulary
        nlp_config.embedding_dim = 128;           // 128-dim embeddings
        nlp_config.max_sequence_length = 32;      // 32 token context
        nlp_config.use_attention_synapses = true;
        nlp_config.use_neuromodulated_synapses = true;

        // Configure base network (required for neural_network_create)
        nlp_config.network_config.num_neurons = 256;  // Small NLP network
        nlp_config.network_config.input_size = nlp_config.embedding_dim;
        nlp_config.network_config.output_size = nlp_config.embedding_dim;
        nlp_config.network_config.enable_stdp = true;
        nlp_config.network_config.enable_hebbian = false;
        nlp_config.network_config.enable_oja = false;
        nlp_config.network_config.enable_homeostasis = false;
        nlp_config.network_config.learning_rate = 0.01F;

        // Configure attention (required for multihead_attention_create)
        nlp_config.attention_config.num_heads = brain->config.num_attention_heads > 0 ? brain->config.num_attention_heads : 4;
        nlp_config.attention_config.input_dim = nlp_config.embedding_dim;
        nlp_config.attention_config.output_dim = nlp_config.embedding_dim;
        nlp_config.attention_config.sequence_length = nlp_config.max_sequence_length;
        nlp_config.attention_config.use_thalamic_gate = false;
        nlp_config.attention_config.use_salience_weighting = false;
        nlp_config.attention_config.gate_bias = 0.5F;

        // Configure neuromodulators (required for neuromodulator_system_create)
        nlp_config.neuromod_config.baseline_dopamine = 0.2F;
        nlp_config.neuromod_config.baseline_serotonin = 0.2F;
        nlp_config.neuromod_config.baseline_acetylcholine = 0.2F;
        nlp_config.neuromod_config.baseline_norepinephrine = 0.2F;
        nlp_config.neuromod_config.dopamine_decay = 2.0F;
        nlp_config.neuromod_config.serotonin_decay = 10.0F;
        nlp_config.neuromod_config.acetylcholine_decay = 0.5F;
        nlp_config.neuromod_config.norepinephrine_decay = 3.0F;
        nlp_config.neuromod_config.reward_dopamine_gain = 0.5F;
        nlp_config.neuromod_config.threat_norepinephrine_gain = 0.7F;
        nlp_config.neuromod_config.salience_acetylcholine_gain = 0.6F;
        nlp_config.neuromod_config.punishment_serotonin_gain = 0.4F;
        nlp_config.neuromod_config.enable_volume_transmission = true;
        nlp_config.neuromod_config.diffusion_rate = 0.1F;

        brain->nlp_network = nlp_network_create(&nlp_config);
        if (!brain->nlp_network) {
            set_error("Failed to create NLP network");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "unknown: brain->nlp_network is NULL");
            return false;
        }
    }

    return true;
}


/**
 * WHAT: Initialize pink noise neuromodulation subsystem
 * WHY:  Enable 1/f noise-modulated dopamine/serotonin for exploration-exploitation balance
 * HOW:  Create pink noise neuromodulator if config flag is set
 *
 * BIOLOGICAL MOTIVATION:
 * - Dopamine neurons exhibit 1/f noise in firing patterns (Montague et al., 2004)
 * - Serotonin fluctuations follow pink spectrum (Cools et al., 2008)
 * - Multi-timescale correlations enable context-dependent learning
 *
 * INTEGRATION:
 * - Modulates learning rates via dopamine
 * - Scales attention via acetylcholine
 * - Enables exploration via pink noise
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 2.7.0 Phase 8.6
 * @author NIMCP Development Team
 * @date 2025-11-08
 */
bool nimcp_brain_factory_init_pink_noise_subsystem(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_pink_noise_subsystem: brain is NULL");

            return false;
    }

    // Check if already initialized
    if (brain->pink_noise) {
        return true;  // Already initialized
    }

    // Check if pink noise is enabled
    if (!brain->config.enable_pink_noise) {
        return true;  // Not enabled, not an error
    }

    // Create pink noise neuromodulator with default configuration
    neuromod_pink_config_t pink_config = neuromod_pink_default_config();

    // Adjust baselines for brain-level processing
    pink_config.dopamine_baseline = 0.3F;      // Moderate baseline for learning
    pink_config.serotonin_baseline = 0.4F;     // Moderate baseline for stability
    pink_config.acetylcholine_baseline = 0.5F; // Moderate baseline for attention
    pink_config.norepinephrine_baseline = 0.2F;// Lower baseline for arousal

    // Configure noise amplitudes for exploration-exploitation balance
    pink_config.dopamine_noise_amplitude = 0.15F;      // 15% noise for exploration
    pink_config.serotonin_noise_amplitude = 0.08F;     // 8% noise for stability modulation
    pink_config.acetylcholine_noise_amplitude = 0.20F; // 20% noise for dynamic attention
    pink_config.norepinephrine_noise_amplitude = 0.10F;// 10% noise for arousal variation

    brain->pink_noise = neuromod_pink_create(&pink_config);
    if (!brain->pink_noise) {
        set_error("Failed to create pink noise neuromodulator");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_pink_noise_subsystem: brain->pink_noise is NULL");
        return false;
    }

    return true;
}
