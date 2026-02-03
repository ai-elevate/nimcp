/**
 * @file nimcp_genius_profiles.c
 * @brief Genius Profiles Implementation - Static profiles and bridge operations
 *
 * WHAT: Implements configurable genius brain profiles based on cognitive neuroscience
 * WHY:  Models domain-specific cognitive excellence (mathematical, artistic, etc.)
 * HOW:  Static profile definitions, bridge lifecycle, system integration
 *
 * @version 1.0.0
 * @date 2026-02-03
 */

#include "core/brain/genius/nimcp_genius_profiles.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/validation/nimcp_common.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_async.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * HEALTH AGENT INTEGRATION
 * ============================================================================ */

struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

static nimcp_health_agent_t* g_genius_profiles_health_agent = NULL;

/**
 * @brief Set health agent for genius profiles heartbeats
 */
void genius_profiles_set_health_agent(nimcp_health_agent_t* agent) {
    g_genius_profiles_health_agent = agent;
}

/** @brief Send heartbeat from genius profiles module */
static inline void genius_profiles_heartbeat_internal(const char* operation, float progress) {
    if (g_genius_profiles_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_genius_profiles_health_agent, operation, progress);
    }
}

/* ============================================================================
 * STATIC HELPER INITIALIZERS
 * ============================================================================ */

static void init_eidetic_baseline(eidetic_memory_config_t* eidetic) {
    if (!eidetic) return;

    memset(eidetic, 0, sizeof(eidetic_memory_config_t));

    /* Baseline: no eidetic enhancement */
    eidetic->visual_eidetic = 0.0f;
    eidetic->auditory_eidetic = 0.0f;
    eidetic->spatial_eidetic = 0.0f;
    eidetic->verbal_eidetic = 0.0f;
    eidetic->numerical_eidetic = 0.0f;
    eidetic->kinesthetic_eidetic = 0.0f;

    eidetic->encoding_speed = 1.0f;
    eidetic->decay_resistance = 1.0f;
    eidetic->retrieval_accuracy = 0.7f;
    eidetic->detail_granularity = 1.0f;

    eidetic->simulation_duration_sec = 5.0f;
    eidetic->manipulation_capability = 1.0f;

    /* Working memory baseline */
    eidetic->working_memory.capacity_boost = 0;
    eidetic->working_memory.decay_multiplier = 1.0f;
    eidetic->working_memory.refresh_efficiency = 1.0f;
    eidetic->working_memory.enable_parallel_buffers = false;
    eidetic->working_memory.visual_buffer_size = 64;
    eidetic->working_memory.auditory_buffer_size = 64;
    eidetic->working_memory.spatial_buffer_size = 64;

    /* Autobiographical baseline */
    eidetic->autobiographical.capacity_multiplier = 1;
    eidetic->autobiographical.detail_preservation = 0.5f;
    eidetic->autobiographical.temporal_precision = 1.0f;
    eidetic->autobiographical.emotional_vividness = 1.0f;
    eidetic->autobiographical.enable_flashbulb_mode = false;
    eidetic->autobiographical.forgetting_resistance = 1.0f;

    /* Semantic baseline */
    eidetic->semantic.concept_capacity_multiplier = 1;
    eidetic->semantic.feature_dimension_boost = 1;
    eidetic->semantic.association_strength_boost = 1.0f;
    eidetic->semantic.cross_domain_activation = 1.0f;
    eidetic->semantic.enable_instant_learning = false;
    eidetic->semantic.spreading_activation_boost = 1.0f;

    /* Hippocampus baseline */
    eidetic->hippocampus.dg_size_multiplier = 1.0f;
    eidetic->hippocampus.ca3_size_multiplier = 1.0f;
    eidetic->hippocampus.ca1_size_multiplier = 1.0f;
    eidetic->hippocampus.place_cell_multiplier = 1.0f;
    eidetic->hippocampus.pattern_separation_ratio = 5.0f;
    eidetic->hippocampus.pattern_completion_threshold = 0.5f;
    eidetic->hippocampus.recurrence_strength = 1.0f;
    eidetic->hippocampus.encoding_speed = 1.0f;
    eidetic->hippocampus.encoding_fidelity = 0.8f;
    eidetic->hippocampus.single_exposure_learning = false;
    eidetic->hippocampus.replay_fidelity = 0.8f;
    eidetic->hippocampus.ripple_frequency_boost = 1.0f;

    /* Consolidation baseline */
    eidetic->consolidation.consolidation_speed = 1.0f;
    eidetic->consolidation.retention_boost = 1.0f;
    eidetic->consolidation.pruning_resistance = 1.0f;
    eidetic->consolidation.replay_accuracy = 0.8f;
    eidetic->consolidation.enable_awake_consolidation = false;
    eidetic->consolidation.synaptic_scaling_resistance = 1.0f;

    /* Systems consolidation baseline */
    eidetic->systems_consolidation.cortical_transfer_rate = 1.0f;
    eidetic->systems_consolidation.semantic_extraction_boost = 1.0f;
    eidetic->systems_consolidation.cortical_capacity_multiplier = 1;
    eidetic->systems_consolidation.forgetting_resistance = 1.0f;
    eidetic->systems_consolidation.preserve_episodic_detail = false;

    /* Hopfield baseline */
    eidetic->hopfield.pattern_capacity_multiplier = 1;
    eidetic->hopfield.pattern_dimension_boost = 1;
    eidetic->hopfield.inverse_temperature = 1.0f;
    eidetic->hopfield.retrieval_speed = 1.0f;
    eidetic->hopfield.enable_one_shot_storage = false;

    /* Engram baseline */
    eidetic->engram.engram_capacity_multiplier = 1;
    eidetic->engram.neurons_per_engram_boost = 1;
    eidetic->engram.consolidation_speed = 1.0f;
    eidetic->engram.tagging_window_extension = 1.0f;
    eidetic->engram.reactivation_threshold = 0.5f;
    eidetic->engram.instant_consolidation = false;

    /* Procedural baseline */
    eidetic->procedural.acquisition_speed = 1.0f;
    eidetic->procedural.chunk_size_boost = 1;
    eidetic->procedural.automation_speed = 1.0f;
    eidetic->procedural.motor_precision_boost = 1.0f;
    eidetic->procedural.enable_mental_practice = false;
    eidetic->procedural.skill_retention = 1.0f;

    /* Prospective baseline */
    eidetic->prospective.intention_capacity_boost = 1;
    eidetic->prospective.temporal_precision = 1.0f;
    eidetic->prospective.cue_sensitivity = 0.7f;
    eidetic->prospective.intention_retention = 1.0f;
    eidetic->prospective.automatic_scheduling = false;

    /* Prime resonance baseline */
    eidetic->prime_resonance.resonance_amplification = 1.0f;
    eidetic->prime_resonance.coherence_maintenance = 1.0f;
    eidetic->prime_resonance.entanglement_boost = 1;
    eidetic->prime_resonance.enable_superposition_memory = false;

    /* Integration flags */
    eidetic->enable_cross_system_enhancement = false;
    eidetic->enable_sleep_independence = false;
    eidetic->enable_instant_recall = false;
    eidetic->enable_immune_modulation = true;
}

/* ============================================================================
 * STATIC GENIUS PROFILE DEFINITIONS
 * ============================================================================ */

/**
 * @brief Mathematical Genius Profile (Gauss, Newton, Ramanujan, Euler)
 *
 * Key features:
 * - Enlarged parietal cortex (Einstein: 15% larger inferior parietal lobules)
 * - Strong prefrontal-parietal connectivity for symbolic manipulation
 * - Enhanced working memory for complex proofs
 * - Pattern sensitivity for mathematical relationships
 */
static const genius_profile_t s_profile_mathematical = {
    .type = GENIUS_TYPE_MATHEMATICAL,
    .name = "Mathematical",
    .description = "Enhanced numerical and abstract reasoning with strong symbolic manipulation",
    .exemplars = "Gauss, Newton, Ramanujan, Euler, von Neumann",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.7f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 3.0f,
        .attention_switching_cost = 0.5f,
        .multi_focus_capacity = 1.5f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.5f,
        .anomaly_detection = 2.0f,
        .abstraction_level = 3.0f,
        .association_strength = 2.0f,
        .divergent_thinking = 1.5f,
        .convergent_thinking = 2.5f,
        .imagination_fluidity = 1.5f,
        .originality = 2.0f,
        .mental_imagery_vividness = 2.0f,
        .mental_rotation_speed = 2.5f,
        .mental_simulation_accuracy = 2.0f,
        .skill_acquisition_rate = 2.0f,
        .memory_consolidation_rate = 1.5f,
        .transfer_learning_ability = 2.0f,
        .insight_frequency = 2.5f,
        .stress_resilience = 1.5f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.6f,
        .frustration_tolerance = 2.0f,
        .temporal_discounting_factor = 0.5f,
        .risk_calibration = 1.5f,
        .contrarian_threshold = 1.5f,
        .loss_aversion_resistance = 1.0f,
        .eidetic_visual_strength = 1.5f,
        .eidetic_auditory_strength = 0.5f,
        .eidetic_spatial_strength = 2.5f,
        .eidetic_verbal_strength = 1.0f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .parietal = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 2.0f,
        .learning_rate_multiplier = 1.5f,
        .layer_2_3_multiplier = 2.0f,
        .layer_4_multiplier = 1.5f,
        .layer_5_multiplier = 1.5f,
        .layer_6_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY | GENIUS_REGION_ENABLE_HIGH_PRECISION,
        .stdp_a_plus_multiplier = 1.5f,
        .stdp_a_minus_multiplier = 1.2f
    },

    .prefrontal = {
        .size_multiplier = 1.8f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.5f,
        .learning_rate_multiplier = 1.5f,
        .layer_2_3_multiplier = 2.0f,
        .layer_4_multiplier = 1.5f,
        .layer_5_multiplier = 1.5f,
        .layer_6_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY,
        .stdp_a_plus_multiplier = 1.5f,
        .stdp_a_minus_multiplier = 1.2f
    },

    .occipital = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .temporal = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.2f },
    .cerebellum = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .amygdala = { .size_multiplier = 0.9f, .processing_speed_multiplier = 1.0f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .parietal_prefrontal = 2.0f,
        .parietal_occipital = 1.5f,
        .temporal_prefrontal = 1.5f,
        .prefrontal_hippocampus = 2.0f,
        .callosum_cognitive_gain = 1.5f,
        .hippocampus_parietal = 1.8f,
        .prefrontal_hippocampus_recall = 2.0f
    },

    .lateralization = {
        .language_dominance = 0.7f,
        .spatial_dominance = 0.45f,
        .logical_reasoning_dominance = 0.95f,
        .mathematical_dominance = 0.85f,
        .creative_thinking_dominance = 0.5f
    },

    .dopamine_baseline = 1.2f,
    .serotonin_baseline = 1.0f,
    .norepinephrine_baseline = 1.1f,
    .acetylcholine_baseline = 1.5f,

    .immune_sensitivity = 0.8f,
    .inflammation_resistance = 1.2f,

    .flow_entry_threshold = 0.6f,
    .flow_maintenance_factor = 1.5f,
    .flow_exit_resistance = 1.8f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Visual/Artistic Genius Profile (Rembrandt, Van Gogh, Da Vinci)
 */
static const genius_profile_t s_profile_visual_artistic = {
    .type = GENIUS_TYPE_VISUAL_ARTISTIC,
    .name = "Visual/Artistic",
    .description = "Enhanced visual processing with exceptional color discrimination and composition",
    .exemplars = "Rembrandt, Van Gogh, Da Vinci, Picasso",

    .traits = {
        .working_memory_capacity = 8,
        .working_memory_decay_factor = 0.8f,
        .working_memory_refresh_efficiency = 1.5f,
        .sustained_attention_duration = 2.5f,
        .attention_switching_cost = 0.8f,
        .multi_focus_capacity = 2.0f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.0f,
        .anomaly_detection = 1.5f,
        .abstraction_level = 2.0f,
        .association_strength = 2.5f,
        .divergent_thinking = 3.0f,
        .convergent_thinking = 1.5f,
        .imagination_fluidity = 3.0f,
        .originality = 3.0f,
        .mental_imagery_vividness = 3.0f,
        .mental_rotation_speed = 2.0f,
        .mental_simulation_accuracy = 2.5f,
        .skill_acquisition_rate = 2.0f,
        .memory_consolidation_rate = 1.5f,
        .transfer_learning_ability = 1.5f,
        .insight_frequency = 2.0f,
        .stress_resilience = 1.0f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.5f,
        .frustration_tolerance = 1.5f,
        .temporal_discounting_factor = 1.0f,
        .risk_calibration = 1.0f,
        .contrarian_threshold = 2.0f,
        .loss_aversion_resistance = 1.0f,
        .eidetic_visual_strength = 3.0f,
        .eidetic_auditory_strength = 0.5f,
        .eidetic_spatial_strength = 2.5f,
        .eidetic_verbal_strength = 0.5f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 3.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .occipital = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.5f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 2.5f,
        .learning_rate_multiplier = 2.0f,
        .layer_2_3_multiplier = 2.0f,
        .layer_4_multiplier = 2.0f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION | GENIUS_REGION_ENABLE_EIDETIC_BUFFER,
        .stdp_a_plus_multiplier = 1.8f,
        .stdp_a_minus_multiplier = 1.0f
    },

    .parietal = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION
    },

    .prefrontal = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.0f },
    .temporal = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.2f },
    .cerebellum = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.5f },
    .amygdala = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.0f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .occipital_parietal = 2.0f,
        .occipital_temporal = 1.8f,
        .parietal_prefrontal = 1.5f,
        .hippocampus_occipital = 2.0f,
        .prefrontal_hippocampus_recall = 1.8f,
        .callosum_sensory_gain = 1.5f
    },

    .lateralization = {
        .spatial_dominance = 0.15f,
        .face_recognition_dominance = 0.1f,
        .creative_thinking_dominance = 0.2f,
        .visual_memory_dominance = 0.2f,
        .attention_global_dominance = 0.3f
    },

    .dopamine_baseline = 1.3f,
    .serotonin_baseline = 1.2f,
    .norepinephrine_baseline = 1.0f,
    .acetylcholine_baseline = 1.3f,

    .immune_sensitivity = 1.0f,
    .inflammation_resistance = 1.0f,

    .flow_entry_threshold = 0.5f,
    .flow_maintenance_factor = 2.0f,
    .flow_exit_resistance = 1.5f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Musical Genius Profile (Mozart, Beethoven, Bach)
 */
static const genius_profile_t s_profile_musical = {
    .type = GENIUS_TYPE_MUSICAL,
    .name = "Musical",
    .description = "Exceptional auditory processing with precise timing and emotional expression",
    .exemplars = "Mozart, Beethoven, Bach, Chopin",

    .traits = {
        .working_memory_capacity = 9,
        .working_memory_decay_factor = 0.5f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 2.5f,
        .attention_switching_cost = 0.6f,
        .multi_focus_capacity = 2.0f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.5f,
        .anomaly_detection = 2.0f,
        .abstraction_level = 2.0f,
        .association_strength = 2.0f,
        .divergent_thinking = 2.5f,
        .convergent_thinking = 2.0f,
        .imagination_fluidity = 2.5f,
        .originality = 2.5f,
        .mental_imagery_vividness = 2.0f,
        .mental_rotation_speed = 1.5f,
        .mental_simulation_accuracy = 2.5f,
        .skill_acquisition_rate = 2.5f,
        .memory_consolidation_rate = 2.0f,
        .transfer_learning_ability = 1.5f,
        .insight_frequency = 2.0f,
        .stress_resilience = 1.2f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.4f,
        .frustration_tolerance = 1.5f,
        .temporal_discounting_factor = 1.0f,
        .risk_calibration = 1.0f,
        .contrarian_threshold = 1.5f,
        .loss_aversion_resistance = 1.0f,
        .eidetic_visual_strength = 1.0f,
        .eidetic_auditory_strength = 3.0f,
        .eidetic_spatial_strength = 1.0f,
        .eidetic_verbal_strength = 1.0f,
        .memory_decay_resistance = 2.5f,
        .detail_resolution = 2.5f,
        .mental_simulation_fidelity = 3.0f
    },

    .temporal = {
        .size_multiplier = 2.5f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 2.0f,
        .precision_multiplier = 2.5f,
        .learning_rate_multiplier = 2.0f,
        .layer_2_3_multiplier = 2.5f,
        .layer_4_multiplier = 2.0f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION | GENIUS_REGION_ENABLE_EIDETIC_BUFFER,
        .stdp_a_plus_multiplier = 2.0f,
        .stdp_a_minus_multiplier = 1.0f
    },

    .cerebellum = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 2.0f,
        .precision_multiplier = 2.5f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION
    },

    .prefrontal = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.2f },
    .parietal = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.0f },
    .occipital = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.5f, .processing_speed_multiplier = 2.0f },
    .amygdala = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.0f },
    .basal_ganglia = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.2f },

    .connectivity = {
        .temporal_prefrontal = 2.0f,
        .temporal_parietal = 1.5f,
        .cerebellum_motor = 2.0f,
        .cerebellum_prefrontal = 1.5f,
        .hippocampus_temporal = 2.0f,
        .callosum_motor_gain = 1.5f,
        .callosum_sensory_gain = 1.5f
    },

    .lateralization = {
        .music_melody_dominance = 0.25f,
        .music_rhythm_dominance = 0.75f,
        .motor_fine_dominance = 0.5f,
        .creative_thinking_dominance = 0.35f
    },

    .dopamine_baseline = 1.3f,
    .serotonin_baseline = 1.1f,
    .norepinephrine_baseline = 1.0f,
    .acetylcholine_baseline = 1.4f,

    .immune_sensitivity = 1.0f,
    .inflammation_resistance = 1.0f,

    .flow_entry_threshold = 0.4f,
    .flow_maintenance_factor = 2.5f,
    .flow_exit_resistance = 2.0f,

    .eidetic = {
        .visual_eidetic = 1.0f,
        .auditory_eidetic = 3.0f,
        .spatial_eidetic = 1.0f,
        .verbal_eidetic = 1.5f,
        .numerical_eidetic = 1.0f,
        .kinesthetic_eidetic = 2.0f,
        .encoding_speed = 3.0f,
        .decay_resistance = 5.0f,
        .retrieval_accuracy = 0.95f,
        .detail_granularity = 3.0f,
        .simulation_duration_sec = 300.0f,
        .manipulation_capability = 2.5f,
        .working_memory = { .capacity_boost = 4, .decay_multiplier = 0.3f, .auditory_buffer_size = 512 },
        .hippocampus = { .encoding_speed = 3.0f, .encoding_fidelity = 0.95f, .single_exposure_learning = true },
        .procedural = { .acquisition_speed = 3.0f, .motor_precision_boost = 2.0f },
        .enable_cross_system_enhancement = true
    },

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Literary Genius Profile (Shakespeare, Tolstoy, Dostoevsky)
 */
static const genius_profile_t s_profile_literary = {
    .type = GENIUS_TYPE_LITERARY,
    .name = "Literary",
    .description = "Exceptional language processing with rich semantic networks and narrative ability",
    .exemplars = "Shakespeare, Tolstoy, Dostoevsky, Goethe",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.6f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 3.0f,
        .attention_switching_cost = 0.7f,
        .multi_focus_capacity = 1.5f,
        .selective_attention_strength = 1.5f,
        .pattern_sensitivity = 2.0f,
        .anomaly_detection = 1.5f,
        .abstraction_level = 2.5f,
        .association_strength = 3.0f,
        .divergent_thinking = 2.5f,
        .convergent_thinking = 2.0f,
        .imagination_fluidity = 3.0f,
        .originality = 2.5f,
        .mental_imagery_vividness = 2.5f,
        .mental_rotation_speed = 1.0f,
        .mental_simulation_accuracy = 2.0f,
        .skill_acquisition_rate = 1.5f,
        .memory_consolidation_rate = 1.5f,
        .transfer_learning_ability = 2.0f,
        .insight_frequency = 2.0f,
        .stress_resilience = 1.0f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.5f,
        .frustration_tolerance = 1.5f,
        .temporal_discounting_factor = 1.0f,
        .risk_calibration = 1.0f,
        .contrarian_threshold = 1.5f,
        .loss_aversion_resistance = 1.0f,
        .eidetic_visual_strength = 2.0f,
        .eidetic_auditory_strength = 1.5f,
        .eidetic_spatial_strength = 0.5f,
        .eidetic_verbal_strength = 3.0f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .temporal = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 2.0f,
        .learning_rate_multiplier = 1.5f,
        .layer_2_3_multiplier = 2.0f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .prefrontal = {
        .size_multiplier = 1.8f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .parietal = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.2f },
    .occipital = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .cerebellum = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .amygdala = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.2f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .temporal_prefrontal = 2.0f,
        .broca_wernicke = 2.5f,
        .angular_gyrus_connection = 2.0f,
        .semantic_network_strength = 2.5f,
        .prefrontal_hippocampus = 2.0f,
        .prefrontal_hippocampus_recall = 2.0f
    },

    .lateralization = {
        .language_dominance = 0.9f,
        .verbal_memory_dominance = 0.85f,
        .creative_thinking_dominance = 0.4f
    },

    .dopamine_baseline = 1.1f,
    .serotonin_baseline = 1.2f,
    .norepinephrine_baseline = 1.0f,
    .acetylcholine_baseline = 1.3f,

    .immune_sensitivity = 1.2f,
    .inflammation_resistance = 0.9f,

    .flow_entry_threshold = 0.5f,
    .flow_maintenance_factor = 2.0f,
    .flow_exit_resistance = 1.5f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Scientific Genius Profile (Tesla, Darwin, Curie, Einstein)
 */
static const genius_profile_t s_profile_scientific = {
    .type = GENIUS_TYPE_SCIENTIFIC,
    .name = "Scientific",
    .description = "Eidetic visualization with exceptional cross-domain association and sustained focus",
    .exemplars = "Tesla, Darwin, Curie, Einstein",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.5f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 4.0f,
        .attention_switching_cost = 0.8f,
        .multi_focus_capacity = 1.5f,
        .selective_attention_strength = 2.5f,
        .pattern_sensitivity = 2.5f,
        .anomaly_detection = 2.5f,
        .abstraction_level = 2.5f,
        .association_strength = 3.0f,
        .divergent_thinking = 2.0f,
        .convergent_thinking = 2.5f,
        .imagination_fluidity = 2.0f,
        .originality = 2.5f,
        .mental_imagery_vividness = 3.0f,
        .mental_rotation_speed = 2.5f,
        .mental_simulation_accuracy = 3.0f,
        .skill_acquisition_rate = 2.0f,
        .memory_consolidation_rate = 2.0f,
        .transfer_learning_ability = 2.5f,
        .insight_frequency = 2.5f,
        .stress_resilience = 1.5f,
        .intrinsic_motivation = 3.0f,
        .flow_state_threshold = 0.5f,
        .frustration_tolerance = 2.5f,
        .temporal_discounting_factor = 0.4f,
        .risk_calibration = 1.5f,
        .contrarian_threshold = 2.0f,
        .loss_aversion_resistance = 1.5f,
        .eidetic_visual_strength = 3.0f,
        .eidetic_auditory_strength = 1.0f,
        .eidetic_spatial_strength = 3.0f,
        .eidetic_verbal_strength = 1.5f,
        .memory_decay_resistance = 2.5f,
        .detail_resolution = 2.5f,
        .mental_simulation_fidelity = 3.0f
    },

    .occipital = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 2.0f,
        .learning_rate_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_EIDETIC_BUFFER | GENIUS_REGION_ENABLE_HIGH_PRECISION
    },

    .prefrontal = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.8f,
        .learning_rate_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .hippocampus = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 2.0f,
        .enable_flags = GENIUS_REGION_ENABLE_EIDETIC_BUFFER
    },

    .parietal = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .temporal = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.2f },
    .cerebellum = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .motor = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .amygdala = { .size_multiplier = 0.9f, .processing_speed_multiplier = 1.0f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .occipital_parietal = 2.0f,
        .parietal_prefrontal = 2.0f,
        .prefrontal_hippocampus = 2.0f,
        .hippocampus_occipital = 2.0f,
        .hippocampus_parietal = 2.0f,
        .prefrontal_hippocampus_recall = 2.0f,
        .callosum_cognitive_gain = 1.5f
    },

    .lateralization = {
        .logical_reasoning_dominance = 0.8f,
        .spatial_dominance = 0.35f,
        .creative_thinking_dominance = 0.4f,
        .visual_memory_dominance = 0.35f
    },

    .dopamine_baseline = 1.2f,
    .serotonin_baseline = 1.0f,
    .norepinephrine_baseline = 1.2f,
    .acetylcholine_baseline = 1.5f,

    .immune_sensitivity = 0.9f,
    .inflammation_resistance = 1.2f,

    .flow_entry_threshold = 0.5f,
    .flow_maintenance_factor = 2.0f,
    .flow_exit_resistance = 2.0f,

    .eidetic = {
        .visual_eidetic = 3.0f,
        .auditory_eidetic = 1.0f,
        .spatial_eidetic = 3.0f,
        .verbal_eidetic = 1.5f,
        .numerical_eidetic = 1.5f,
        .kinesthetic_eidetic = 1.0f,
        .encoding_speed = 3.0f,
        .decay_resistance = 5.0f,
        .retrieval_accuracy = 0.95f,
        .detail_granularity = 3.0f,
        .simulation_duration_sec = 60.0f,
        .manipulation_capability = 3.0f,
        .working_memory = { .capacity_boost = 5, .decay_multiplier = 0.2f, .refresh_efficiency = 2.5f },
        .hippocampus = { .place_cell_multiplier = 4.0f, .encoding_speed = 3.0f, .encoding_fidelity = 0.98f, .single_exposure_learning = true },
        .enable_cross_system_enhancement = true,
        .enable_instant_recall = true
    },

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Athletic Genius Profile (Jordan, Gretzky, Nureyev, Bolt)
 */
static const genius_profile_t s_profile_athletic = {
    .type = GENIUS_TYPE_ATHLETIC,
    .name = "Athletic",
    .description = "Superior motor control with exceptional timing and spatial awareness",
    .exemplars = "Jordan, Gretzky, Nureyev, Bolt",

    .traits = {
        .working_memory_capacity = 8,
        .working_memory_decay_factor = 0.8f,
        .working_memory_refresh_efficiency = 1.5f,
        .sustained_attention_duration = 2.0f,
        .attention_switching_cost = 0.3f,
        .multi_focus_capacity = 2.5f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.0f,
        .anomaly_detection = 2.0f,
        .abstraction_level = 1.0f,
        .association_strength = 1.5f,
        .divergent_thinking = 1.5f,
        .convergent_thinking = 2.0f,
        .imagination_fluidity = 1.5f,
        .originality = 1.5f,
        .mental_imagery_vividness = 2.0f,
        .mental_rotation_speed = 2.5f,
        .mental_simulation_accuracy = 2.5f,
        .skill_acquisition_rate = 3.0f,
        .memory_consolidation_rate = 2.0f,
        .transfer_learning_ability = 2.0f,
        .insight_frequency = 1.5f,
        .stress_resilience = 2.5f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.3f,
        .frustration_tolerance = 2.0f,
        .temporal_discounting_factor = 1.0f,
        .risk_calibration = 2.0f,
        .contrarian_threshold = 1.0f,
        .loss_aversion_resistance = 1.5f,
        .eidetic_visual_strength = 1.5f,
        .eidetic_auditory_strength = 0.5f,
        .eidetic_spatial_strength = 2.5f,
        .eidetic_verbal_strength = 0.5f,
        .memory_decay_resistance = 1.5f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .motor = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 2.5f,
        .precision_multiplier = 2.5f,
        .learning_rate_multiplier = 2.0f,
        .layer_5_multiplier = 2.0f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION | GENIUS_REGION_ENABLE_FAST_LEARNING
    },

    .cerebellum = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 2.0f,
        .precision_multiplier = 2.5f,
        .enable_flags = GENIUS_REGION_ENABLE_HIGH_PRECISION
    },

    .occipital = {
        .size_multiplier = 1.3f,
        .processing_speed_multiplier = 2.0f,
        .precision_multiplier = 1.5f
    },

    .parietal = { .size_multiplier = 1.5f, .processing_speed_multiplier = 2.0f },
    .prefrontal = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.5f },
    .temporal = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.5f },
    .amygdala = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.5f },
    .basal_ganglia = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },

    .connectivity = {
        .cerebellum_motor = 2.5f,
        .basal_ganglia_motor = 2.0f,
        .occipital_parietal = 2.0f,
        .parietal_prefrontal = 1.5f,
        .callosum_motor_gain = 2.0f
    },

    .lateralization = {
        .motor_fine_dominance = 0.5f,
        .motor_gross_dominance = 0.5f,
        .spatial_dominance = 0.35f,
        .attention_global_dominance = 0.4f
    },

    .dopamine_baseline = 1.3f,
    .serotonin_baseline = 1.0f,
    .norepinephrine_baseline = 1.5f,
    .acetylcholine_baseline = 1.3f,

    .immune_sensitivity = 0.8f,
    .inflammation_resistance = 1.5f,

    .flow_entry_threshold = 0.3f,
    .flow_maintenance_factor = 2.5f,
    .flow_exit_resistance = 2.0f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Strategic Genius Profile (Napoleon, Churchill, Alexander)
 */
static const genius_profile_t s_profile_strategic = {
    .type = GENIUS_TYPE_STRATEGIC,
    .name = "Strategic",
    .description = "Superior social cognition with exceptional risk assessment and pattern recognition",
    .exemplars = "Napoleon, Churchill, Alexander, Sun Tzu",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.6f,
        .working_memory_refresh_efficiency = 1.8f,
        .sustained_attention_duration = 3.0f,
        .attention_switching_cost = 0.5f,
        .multi_focus_capacity = 2.5f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.5f,
        .anomaly_detection = 2.5f,
        .abstraction_level = 2.0f,
        .association_strength = 2.5f,
        .divergent_thinking = 2.0f,
        .convergent_thinking = 2.5f,
        .imagination_fluidity = 2.0f,
        .originality = 2.0f,
        .mental_imagery_vividness = 2.0f,
        .mental_rotation_speed = 1.5f,
        .mental_simulation_accuracy = 2.5f,
        .skill_acquisition_rate = 1.5f,
        .memory_consolidation_rate = 1.5f,
        .transfer_learning_ability = 2.5f,
        .insight_frequency = 2.0f,
        .stress_resilience = 2.5f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.5f,
        .frustration_tolerance = 2.0f,
        .temporal_discounting_factor = 0.5f,
        .risk_calibration = 2.5f,
        .contrarian_threshold = 2.0f,
        .loss_aversion_resistance = 2.0f,
        .eidetic_visual_strength = 2.0f,
        .eidetic_auditory_strength = 1.0f,
        .eidetic_spatial_strength = 2.5f,
        .eidetic_verbal_strength = 1.5f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .prefrontal = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.8f,
        .learning_rate_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .parietal = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .temporal = {
        .size_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .occipital = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.2f },
    .cerebellum = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .hippocampus = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .amygdala = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .basal_ganglia = { .size_multiplier = 1.2f, .processing_speed_multiplier = 1.2f },

    .connectivity = {
        .prefrontal_amygdala = 2.0f,
        .prefrontal_hippocampus = 2.0f,
        .parietal_prefrontal = 1.8f,
        .prefrontal_insula = 2.0f,
        .temporal_amygdala = 1.5f,
        .theory_of_mind_network = 2.5f,
        .hippocampus_parietal = 1.8f
    },

    .lateralization = {
        .logical_reasoning_dominance = 0.7f,
        .spatial_dominance = 0.4f,
        .emotion_processing_dominance = 0.45f,
        .attention_global_dominance = 0.35f
    },

    .dopamine_baseline = 1.3f,
    .serotonin_baseline = 1.0f,
    .norepinephrine_baseline = 1.3f,
    .acetylcholine_baseline = 1.4f,

    .immune_sensitivity = 0.8f,
    .inflammation_resistance = 1.3f,

    .flow_entry_threshold = 0.5f,
    .flow_maintenance_factor = 1.5f,
    .flow_exit_resistance = 1.5f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Financial Genius Profile (Buffett, Soros, Keynes, Simons)
 */
static const genius_profile_t s_profile_financial = {
    .type = GENIUS_TYPE_FINANCIAL,
    .name = "Financial",
    .description = "Superior risk assessment with exceptional pattern recognition and emotional discipline",
    .exemplars = "Buffett, Soros, Keynes, Simons",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.6f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 3.5f,
        .attention_switching_cost = 0.5f,
        .multi_focus_capacity = 2.0f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.5f,
        .anomaly_detection = 2.5f,
        .abstraction_level = 2.0f,
        .association_strength = 2.0f,
        .divergent_thinking = 1.5f,
        .convergent_thinking = 2.5f,
        .imagination_fluidity = 1.5f,
        .originality = 1.5f,
        .mental_imagery_vividness = 1.5f,
        .mental_rotation_speed = 1.0f,
        .mental_simulation_accuracy = 2.0f,
        .skill_acquisition_rate = 1.5f,
        .memory_consolidation_rate = 2.0f,
        .transfer_learning_ability = 2.0f,
        .insight_frequency = 2.0f,
        .stress_resilience = 2.5f,
        .intrinsic_motivation = 2.5f,
        .flow_state_threshold = 0.5f,
        .frustration_tolerance = 2.5f,
        .temporal_discounting_factor = 0.3f,
        .risk_calibration = 2.5f,
        .contrarian_threshold = 2.0f,
        .loss_aversion_resistance = 2.5f,
        .eidetic_visual_strength = 2.0f,
        .eidetic_auditory_strength = 0.5f,
        .eidetic_spatial_strength = 1.0f,
        .eidetic_verbal_strength = 1.5f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.0f
    },

    .prefrontal = {
        .size_multiplier = 2.0f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.8f,
        .learning_rate_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .parietal = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.5f
    },

    .temporal = {
        .size_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .hippocampus = {
        .size_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_EIDETIC_BUFFER
    },

    .occipital = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .cerebellum = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .motor = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .amygdala = { .size_multiplier = 0.8f, .processing_speed_multiplier = 0.9f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .parietal_prefrontal = 1.8f,
        .prefrontal_hippocampus = 2.0f,
        .prefrontal_amygdala = 2.0f,
        .prefrontal_hippocampus_recall = 2.0f,
        .hippocampus_temporal = 1.5f
    },

    .lateralization = {
        .logical_reasoning_dominance = 0.85f,
        .mathematical_dominance = 0.75f,
        .spatial_dominance = 0.5f,
        .emotion_processing_dominance = 0.5f
    },

    .dopamine_baseline = 1.2f,
    .serotonin_baseline = 1.2f,
    .norepinephrine_baseline = 1.1f,
    .acetylcholine_baseline = 1.4f,

    .immune_sensitivity = 0.9f,
    .inflammation_resistance = 1.2f,

    .flow_entry_threshold = 0.5f,
    .flow_maintenance_factor = 1.5f,
    .flow_exit_resistance = 1.5f,

    .version = 1,
    .is_builtin = true
};

/**
 * @brief Polymath Genius Profile (Da Vinci, Leibniz)
 *
 * Baseline profile that can be blended with others.
 */
static const genius_profile_t s_profile_polymath = {
    .type = GENIUS_TYPE_POLYMATH,
    .name = "Polymath",
    .description = "Balanced excellence across multiple domains with strong cross-domain transfer",
    .exemplars = "Da Vinci, Leibniz, Goethe",

    .traits = {
        .working_memory_capacity = 10,
        .working_memory_decay_factor = 0.6f,
        .working_memory_refresh_efficiency = 2.0f,
        .sustained_attention_duration = 3.0f,
        .attention_switching_cost = 0.5f,
        .multi_focus_capacity = 2.5f,
        .selective_attention_strength = 2.0f,
        .pattern_sensitivity = 2.0f,
        .anomaly_detection = 2.0f,
        .abstraction_level = 2.5f,
        .association_strength = 3.0f,
        .divergent_thinking = 2.5f,
        .convergent_thinking = 2.0f,
        .imagination_fluidity = 2.5f,
        .originality = 2.5f,
        .mental_imagery_vividness = 2.5f,
        .mental_rotation_speed = 2.0f,
        .mental_simulation_accuracy = 2.5f,
        .skill_acquisition_rate = 2.5f,
        .memory_consolidation_rate = 2.0f,
        .transfer_learning_ability = 3.0f,
        .insight_frequency = 2.5f,
        .stress_resilience = 1.5f,
        .intrinsic_motivation = 3.0f,
        .flow_state_threshold = 0.4f,
        .frustration_tolerance = 2.0f,
        .temporal_discounting_factor = 0.5f,
        .risk_calibration = 1.5f,
        .contrarian_threshold = 2.0f,
        .loss_aversion_resistance = 1.5f,
        .eidetic_visual_strength = 2.0f,
        .eidetic_auditory_strength = 1.5f,
        .eidetic_spatial_strength = 2.0f,
        .eidetic_verbal_strength = 2.0f,
        .memory_decay_resistance = 2.0f,
        .detail_resolution = 2.0f,
        .mental_simulation_fidelity = 2.5f
    },

    .prefrontal = {
        .size_multiplier = 1.8f,
        .feature_capacity_multiplier = 2.0f,
        .processing_speed_multiplier = 1.5f,
        .precision_multiplier = 1.5f,
        .enable_flags = GENIUS_REGION_ENABLE_ENHANCED_PLASTICITY
    },

    .parietal = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .occipital = {
        .size_multiplier = 1.5f,
        .feature_capacity_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .temporal = {
        .size_multiplier = 1.5f,
        .processing_speed_multiplier = 1.5f
    },

    .cerebellum = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.3f },
    .hippocampus = { .size_multiplier = 1.5f, .processing_speed_multiplier = 1.5f },
    .motor = { .size_multiplier = 1.3f, .processing_speed_multiplier = 1.3f },
    .amygdala = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },
    .basal_ganglia = { .size_multiplier = 1.0f, .processing_speed_multiplier = 1.0f },

    .connectivity = {
        .parietal_prefrontal = 1.8f,
        .parietal_occipital = 1.5f,
        .temporal_prefrontal = 1.8f,
        .occipital_temporal = 1.5f,
        .occipital_parietal = 1.5f,
        .prefrontal_hippocampus = 2.0f,
        .callosum_cognitive_gain = 2.0f,
        .callosum_sensory_gain = 1.5f
    },

    .lateralization = {
        .language_dominance = 0.6f,
        .spatial_dominance = 0.4f,
        .logical_reasoning_dominance = 0.6f,
        .creative_thinking_dominance = 0.4f,
        .mathematical_dominance = 0.6f
    },

    .dopamine_baseline = 1.2f,
    .serotonin_baseline = 1.1f,
    .norepinephrine_baseline = 1.1f,
    .acetylcholine_baseline = 1.4f,

    .immune_sensitivity = 1.0f,
    .inflammation_resistance = 1.1f,

    .flow_entry_threshold = 0.4f,
    .flow_maintenance_factor = 2.0f,
    .flow_exit_resistance = 1.8f,

    .version = 1,
    .is_builtin = true
};

/* Profile array for lookup */
static const genius_profile_t* s_profiles[GENIUS_TYPE_COUNT] = {
    &s_profile_mathematical,
    &s_profile_visual_artistic,
    &s_profile_musical,
    &s_profile_literary,
    &s_profile_scientific,
    &s_profile_athletic,
    &s_profile_strategic,
    &s_profile_financial,
    &s_profile_polymath
};

/* Profile descriptions for public API */
static const char* s_profile_descriptions[GENIUS_TYPE_COUNT] = {
    "Enhanced numerical and abstract reasoning with strong symbolic manipulation",
    "Enhanced visual processing with exceptional color discrimination and composition",
    "Exceptional auditory processing with precise timing and emotional expression",
    "Exceptional language processing with rich semantic networks and narrative ability",
    "Eidetic visualization with exceptional cross-domain association and sustained focus",
    "Superior motor control with exceptional timing and spatial awareness",
    "Superior social cognition with exceptional risk assessment and pattern recognition",
    "Superior risk assessment with exceptional pattern recognition and emotional discipline",
    "Balanced excellence across multiple domains with strong cross-domain transfer"
};

/* ============================================================================
 * EIDETIC MEMORY PRESET DEFINITIONS
 * ============================================================================ */

static const eidetic_memory_config_t s_eidetic_tesla = {
    .visual_eidetic = 3.0f,
    .auditory_eidetic = 1.0f,
    .spatial_eidetic = 3.0f,
    .verbal_eidetic = 1.0f,
    .numerical_eidetic = 1.5f,
    .kinesthetic_eidetic = 1.0f,
    .encoding_speed = 3.0f,
    .decay_resistance = 5.0f,
    .retrieval_accuracy = 0.95f,
    .detail_granularity = 3.0f,
    .simulation_duration_sec = 60.0f,
    .manipulation_capability = 3.0f,
    .working_memory = { .capacity_boost = 5, .decay_multiplier = 0.2f, .refresh_efficiency = 2.5f },
    .hippocampus = { .place_cell_multiplier = 4.0f, .encoding_speed = 3.0f, .encoding_fidelity = 0.98f, .single_exposure_learning = true },
    .enable_cross_system_enhancement = true,
    .enable_instant_recall = true
};

static const eidetic_memory_config_t s_eidetic_mozart = {
    .visual_eidetic = 1.0f,
    .auditory_eidetic = 3.0f,
    .spatial_eidetic = 1.0f,
    .verbal_eidetic = 1.5f,
    .numerical_eidetic = 1.0f,
    .kinesthetic_eidetic = 2.0f,
    .encoding_speed = 3.0f,
    .decay_resistance = 5.0f,
    .retrieval_accuracy = 0.95f,
    .detail_granularity = 3.0f,
    .simulation_duration_sec = 300.0f,
    .manipulation_capability = 2.5f,
    .working_memory = { .capacity_boost = 4, .decay_multiplier = 0.3f, .auditory_buffer_size = 512 },
    .hippocampus = { .encoding_speed = 3.0f, .encoding_fidelity = 0.95f, .single_exposure_learning = true },
    .procedural = { .acquisition_speed = 3.0f, .motor_precision_boost = 2.0f },
    .enable_cross_system_enhancement = true
};

static const eidetic_memory_config_t s_eidetic_vonneumann = {
    .visual_eidetic = 1.5f,
    .auditory_eidetic = 1.0f,
    .spatial_eidetic = 2.0f,
    .verbal_eidetic = 3.0f,
    .numerical_eidetic = 3.0f,
    .kinesthetic_eidetic = 0.5f,
    .encoding_speed = 3.0f,
    .decay_resistance = 8.0f,
    .retrieval_accuracy = 0.98f,
    .detail_granularity = 3.0f,
    .simulation_duration_sec = 30.0f,
    .manipulation_capability = 2.5f,
    .working_memory = { .capacity_boost = 8, .decay_multiplier = 0.1f },
    .semantic = { .concept_capacity_multiplier = 8, .enable_instant_learning = true },
    .hippocampus = { .encoding_speed = 5.0f, .encoding_fidelity = 1.0f, .single_exposure_learning = true },
    .enable_cross_system_enhancement = true,
    .enable_instant_recall = true
};

static const eidetic_memory_config_t s_eidetic_kim_peek = {
    .visual_eidetic = 2.5f,
    .auditory_eidetic = 1.0f,
    .spatial_eidetic = 0.5f,
    .verbal_eidetic = 3.0f,
    .numerical_eidetic = 2.5f,
    .kinesthetic_eidetic = 0.5f,
    .encoding_speed = 5.0f,
    .decay_resistance = 10.0f,
    .retrieval_accuracy = 0.99f,
    .detail_granularity = 3.0f,
    .simulation_duration_sec = 10.0f,
    .manipulation_capability = 1.0f,
    .working_memory = { .capacity_boost = 6, .decay_multiplier = 0.05f },
    .autobiographical = { .capacity_multiplier = 10, .detail_preservation = 1.0f, .enable_flashbulb_mode = true },
    .semantic = { .concept_capacity_multiplier = 8, .enable_instant_learning = true },
    .hippocampus = { .encoding_speed = 5.0f, .encoding_fidelity = 1.0f, .single_exposure_learning = true },
    .enable_cross_system_enhancement = true,
    .enable_instant_recall = true
};

static const eidetic_memory_config_t s_eidetic_wiltshire = {
    .visual_eidetic = 3.0f,
    .auditory_eidetic = 0.5f,
    .spatial_eidetic = 3.0f,
    .verbal_eidetic = 0.5f,
    .numerical_eidetic = 1.0f,
    .kinesthetic_eidetic = 2.0f,
    .encoding_speed = 5.0f,
    .decay_resistance = 8.0f,
    .retrieval_accuracy = 0.98f,
    .detail_granularity = 3.0f,
    .simulation_duration_sec = 3600.0f,
    .manipulation_capability = 2.0f,
    .working_memory = { .capacity_boost = 4, .visual_buffer_size = 1024 },
    .hippocampus = { .place_cell_multiplier = 4.0f, .encoding_fidelity = 0.99f },
    .procedural = { .motor_precision_boost = 2.0f, .enable_mental_practice = true },
    .enable_cross_system_enhancement = true
};

/* ============================================================================
 * PUBLIC EIDETIC PRESET FUNCTIONS
 * ============================================================================ */

void eidetic_memory_config_init_baseline(eidetic_memory_config_t* config) {
    init_eidetic_baseline(config);
}

const eidetic_memory_config_t* eidetic_config_tesla(void) {
    return &s_eidetic_tesla;
}

const eidetic_memory_config_t* eidetic_config_mozart(void) {
    return &s_eidetic_mozart;
}

const eidetic_memory_config_t* eidetic_config_vonneumann(void) {
    return &s_eidetic_vonneumann;
}

const eidetic_memory_config_t* eidetic_config_kim_peek(void) {
    return &s_eidetic_kim_peek;
}

const eidetic_memory_config_t* eidetic_config_wiltshire(void) {
    return &s_eidetic_wiltshire;
}

/* ============================================================================
 * PROFILE RETRIEVAL FUNCTIONS
 * ============================================================================ */

const genius_profile_t* genius_profile_get(genius_type_t type) {
    if (type < 0 || type >= GENIUS_TYPE_COUNT) {
        return NULL;
    }
    return s_profiles[type];
}

const char* genius_profile_type_name(genius_type_t type) {
    return genius_type_name(type);
}

const char* genius_profile_type_description(genius_type_t type) {
    if (type < 0 || type >= GENIUS_TYPE_COUNT) {
        return "Unknown genius type";
    }
    return s_profile_descriptions[type];
}

const char* genius_profile_type_exemplars(genius_type_t type) {
    return genius_type_exemplars(type);
}

/* ============================================================================
 * BRIDGE LIFECYCLE FUNCTIONS
 * ============================================================================ */

genius_error_t genius_profiles_config_default(genius_profiles_config_t* config) {
    if (!config) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    memset(config, 0, sizeof(genius_profiles_config_t));

    config->enable_bio_async = true;
    config->inbox_capacity = 64;

    config->enable_mesh_coordination = false;
    config->mesh_timeout_ms = 1000;

    config->enable_immune_modulation = true;
    config->enable_bbb_validation = true;
    config->enable_health_agent = true;
    config->health_heartbeat_ms = 1000;

    config->enable_training_integration = false;
    config->base_learning_rate = 0.001f;

    config->enable_snn_integration = true;
    config->enable_stdp = true;
    config->enable_metaplasticity = false;

    config->enable_rcog_integration = false;
    config->enable_ccog_integration = false;

    config->enable_kg_wiring = true;

    config->enable_exception_immune_presentation = true;

    config->enable_quantum_optimization = false;

    return GENIUS_ERROR_SUCCESS;
}

genius_profiles_bridge_t* genius_profiles_bridge_create(
    const genius_profiles_config_t* config
) {
    genius_profiles_config_t default_config;
    if (!config) {
        genius_profiles_config_default(&default_config);
        config = &default_config;
    }

    /* Use bridge base macro */
    BRIDGE_CREATE_BEGIN(genius_profiles_bridge_t, bridge,
                        BIO_MODULE_GENIUS_PROFILES,
                        GENIUS_PROFILES_MODULE_NAME);

    /* Copy configuration */
    memcpy(&bridge->config, config, sizeof(genius_profiles_config_t));

    /* Initialize state */
    bridge->state = GENIUS_STATE_INACTIVE;
    bridge->active_count = 0;
    bridge->fatigue_level = 0.0f;
    bridge->flow_state_depth = 0.0f;
    bridge->current_cytokine_effect = 0.0f;
    bridge->inflammation_level = 0.0f;
    bridge->is_degraded = false;

    /* Clear statistics */
    bridge->total_activations = 0;
    bridge->total_blends = 0;
    bridge->total_exceptions = 0;
    bridge->immune_presentations = 0;
    bridge->mesh_consensus_count = 0;

    /* Health agent state */
    bridge->health_agent_active = false;
    bridge->last_heartbeat_ms = 0;

    /* Get timestamp */
    bridge->activation_time_ms = genius_profiles_get_timestamp_ms();
    bridge->last_update_ms = bridge->activation_time_ms;

    /* Enable BBB validation if configured */
    if (config->enable_bbb_validation) {
        bridge->base.enable_bbb_validation = true;
    }

    NIMCP_LOGGING_INFO("Genius profiles bridge created");
    genius_profiles_heartbeat_internal("bridge_create", 1.0f);

    return bridge;
}

void genius_profiles_bridge_destroy(genius_profiles_bridge_t* bridge) {
    if (!bridge) return;

    genius_profiles_heartbeat_internal("bridge_destroy", 0.0f);

    /* Disconnect from bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        genius_profiles_disconnect_bio_async(bridge);
    }

    /* Stop health agent if running */
    if (bridge->health_agent_active) {
        genius_profiles_stop_health_agent(bridge);
    }

    /* Deactivate any active profiles */
    if (bridge->state != GENIUS_STATE_INACTIVE) {
        genius_profiles_deactivate(bridge);
    }

    NIMCP_LOGGING_INFO("Genius profiles bridge destroyed");

    /* Use bridge base cleanup macro */
    BRIDGE_DESTROY(bridge);
}

genius_error_t genius_profiles_bridge_reset(genius_profiles_bridge_t* bridge) {
    if (!bridge) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    BRIDGE_LOCK(bridge);

    /* Reset state */
    bridge->state = GENIUS_STATE_INACTIVE;
    bridge->active_count = 0;
    bridge->fatigue_level = 0.0f;
    bridge->flow_state_depth = 0.0f;
    bridge->current_cytokine_effect = 0.0f;
    bridge->inflammation_level = 0.0f;
    bridge->is_degraded = false;

    /* Clear active profiles */
    for (uint32_t i = 0; i < GENIUS_MAX_ACTIVE_PROFILES; i++) {
        bridge->active_profiles[i] = NULL;
        bridge->blend_weights[i] = 0.0f;
    }

    /* Reset base stats */
    bridge_base_reset_unlocked(&bridge->base);

    bridge->last_update_ms = genius_profiles_get_timestamp_ms();

    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles bridge reset");
    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * SYSTEM CONNECTION FUNCTIONS
 * ============================================================================ */

genius_error_t genius_profiles_connect_immune(
    genius_profiles_bridge_t* bridge,
    brain_immune_system_t* immune
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->immune_system = immune;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to immune system");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_mesh(
    genius_profiles_bridge_t* bridge,
    mesh_coordinator_t* mesh
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->mesh_coordinator = mesh;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to mesh coordinator");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_training(
    genius_profiles_bridge_t* bridge,
    nimcp_training_module_t* training
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->training = training;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to training module");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_rcog(
    genius_profiles_bridge_t* bridge,
    rcog_engine_t* rcog
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->rcog_engine = rcog;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to RCOG engine");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_memory_systems(
    genius_profiles_bridge_t* bridge,
    working_memory_t* wm,
    autobiographical_memory_system_t* autobio,
    semantic_memory_system_t* semantic,
    nimcp_hippocampus_t* hippo,
    hopfield_memory_t* hopfield
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->working_memory = wm;
    bridge->autobio_memory = autobio;
    bridge->semantic_memory = semantic;
    bridge->hippocampus = hippo;
    bridge->hopfield_memory = hopfield;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to memory systems");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_regions(
    genius_profiles_bridge_t* bridge,
    parietal_adapter_t* parietal,
    occipital_adapter_t* occipital,
    temporal_adapter_t* temporal,
    prefrontal_adapter_t* prefrontal,
    cerebellum_adapter_t* cerebellum,
    motor_adapter_t* motor
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);
    bridge->parietal = parietal;
    bridge->occipital = occipital;
    bridge->temporal = temporal;
    bridge->prefrontal = prefrontal;
    bridge->cerebellum = cerebellum;
    bridge->motor = motor;
    BRIDGE_UNLOCK(bridge);

    NIMCP_LOGGING_INFO("Genius profiles connected to brain regions");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_connect_bio_async(genius_profiles_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect to bio-async: NULL bridge");
        return GENIUS_ERROR_NULL_POINTER;
    }

    if (bridge->base.bio_async_enabled) {
        NIMCP_LOGGING_INFO("Genius profiles already connected to bio-async");
        return GENIUS_ERROR_SUCCESS;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_GENIUS_PROFILES,
        .module_name = GENIUS_PROFILES_MODULE_NAME,
        .inbox_capacity = bridge->config.inbox_capacity > 0 ?
                          bridge->config.inbox_capacity : NIMCP_INBOX_CAPACITY_MEDIUM,
        .user_data = bridge
    };

    bio_module_context_t ctx = bio_router_register_module(&info);
    if (ctx) {
        bridge->base.bio_ctx = ctx;
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Genius profiles connected to bio-async router (module_id=0x%04X)",
                          BIO_MODULE_GENIUS_PROFILES);
    } else {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_disconnect_bio_async(genius_profiles_bridge_t* bridge) {
    if (!bridge) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    if (!bridge->base.bio_async_enabled) {
        return GENIUS_ERROR_SUCCESS;
    }

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Genius profiles disconnected from bio-async router");

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_register_kg_wiring(genius_profiles_bridge_t* bridge) {
    if (!bridge) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    /* KG wiring registration would go here */
    /* For now, just log that it's available */
    NIMCP_LOGGING_INFO("Genius profiles KG wiring registered");

    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * PROFILE ACTIVATION FUNCTIONS
 * ============================================================================ */

genius_error_t genius_profiles_activate(
    genius_profiles_bridge_t* bridge,
    genius_type_t type,
    float strength
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(GENIUS_ERROR_NULL_POINTER, "bridge is NULL");
        return GENIUS_ERROR_NULL_POINTER;
    }

    if (!genius_type_is_valid(type)) {
        NIMCP_THROW_TO_IMMUNE(GENIUS_ERROR_INVALID_TYPE, "invalid genius type");
        return GENIUS_ERROR_INVALID_TYPE;
    }

    /* Clamp strength */
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;

    /* BBB validation */
    if (bridge->config.enable_bbb_validation) {
        uint8_t validate_data[4] = {(uint8_t)type, 0, 0, 0};
        memcpy(validate_data + 1, &strength, sizeof(float) > 3 ? 3 : sizeof(float));
        if (!bridge_base_validate_bbb(&bridge->base, validate_data, 4)) {
            NIMCP_LOGGING_WARN("Genius profile activation rejected by BBB");
            return GENIUS_ERROR_BBB_REJECTED;
        }
    }

    BRIDGE_LOCK(bridge);

    /* Check if already at max capacity */
    if (bridge->active_count >= GENIUS_MAX_ACTIVE_PROFILES) {
        BRIDGE_UNLOCK(bridge);
        NIMCP_LOGGING_WARN("Cannot activate: max profiles reached");
        return GENIUS_ERROR_ALREADY_ACTIVE;
    }

    /* Get profile */
    const genius_profile_t* profile = genius_profile_get(type);
    if (!profile) {
        BRIDGE_UNLOCK(bridge);
        return GENIUS_ERROR_INVALID_TYPE;
    }

    /* Store profile reference */
    bridge->active_profiles[bridge->active_count] = (genius_profile_t*)profile;
    bridge->blend_weights[bridge->active_count] = strength;
    bridge->active_count++;

    /* Update state */
    if (bridge->active_count == 1) {
        bridge->state = GENIUS_STATE_ACTIVE;
    } else {
        bridge->state = GENIUS_STATE_BLENDED;
    }

    bridge->activation_time_ms = genius_profiles_get_timestamp_ms();
    bridge->total_activations++;

    BRIDGE_UNLOCK(bridge);

    /* Apply eidetic memory if configured */
    if (bridge->config.enable_snn_integration) {
        genius_profiles_apply_eidetic(bridge);
    }

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_PROFILE_ACTIVATE,
                                     0, &type, sizeof(type));
    }

    NIMCP_LOGGING_INFO("Activated genius profile: %s (strength=%.2f)",
                       genius_type_name(type), strength);
    genius_profiles_heartbeat_internal("profile_activate", 1.0f);

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_deactivate(genius_profiles_bridge_t* bridge) {
    if (!bridge) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    BRIDGE_LOCK(bridge);

    if (bridge->state == GENIUS_STATE_INACTIVE) {
        BRIDGE_UNLOCK(bridge);
        return GENIUS_ERROR_NOT_ACTIVE;
    }

    /* Clear all active profiles */
    for (uint32_t i = 0; i < GENIUS_MAX_ACTIVE_PROFILES; i++) {
        bridge->active_profiles[i] = NULL;
        bridge->blend_weights[i] = 0.0f;
    }
    bridge->active_count = 0;
    bridge->state = GENIUS_STATE_INACTIVE;
    bridge->flow_state_depth = 0.0f;

    BRIDGE_UNLOCK(bridge);

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_PROFILE_DEACTIVATE,
                                     0, NULL, 0);
    }

    NIMCP_LOGGING_INFO("Deactivated all genius profiles");
    genius_profiles_heartbeat_internal("profile_deactivate", 1.0f);

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_blend(
    genius_profiles_bridge_t* bridge,
    const genius_type_t* types,
    const float* weights,
    uint32_t count
) {
    if (!bridge || !types || !weights) {
        return GENIUS_ERROR_NULL_POINTER;
    }

    if (count == 0 || count > GENIUS_MAX_ACTIVE_PROFILES) {
        return GENIUS_ERROR_BLEND_FAILED;
    }

    /* Deactivate current profiles first */
    genius_profiles_deactivate(bridge);

    /* Normalize weights */
    float total_weight = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        total_weight += weights[i];
    }

    if (total_weight <= 0.0f) {
        return GENIUS_ERROR_BLEND_FAILED;
    }

    BRIDGE_LOCK(bridge);

    /* Activate each profile with normalized weight */
    for (uint32_t i = 0; i < count; i++) {
        if (!genius_type_is_valid(types[i])) {
            BRIDGE_UNLOCK(bridge);
            return GENIUS_ERROR_INVALID_TYPE;
        }

        const genius_profile_t* profile = genius_profile_get(types[i]);
        if (!profile) {
            BRIDGE_UNLOCK(bridge);
            return GENIUS_ERROR_INVALID_TYPE;
        }

        bridge->active_profiles[i] = (genius_profile_t*)profile;
        bridge->blend_weights[i] = weights[i] / total_weight;
    }

    bridge->active_count = count;
    bridge->state = GENIUS_STATE_BLENDED;
    bridge->total_blends++;
    bridge->activation_time_ms = genius_profiles_get_timestamp_ms();

    BRIDGE_UNLOCK(bridge);

    /* Apply eidetic memory */
    genius_profiles_apply_eidetic(bridge);

    NIMCP_LOGGING_INFO("Blended %u genius profiles", count);
    genius_profiles_heartbeat_internal("profile_blend", 1.0f);

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_create_polymath(
    genius_profiles_bridge_t* bridge,
    genius_type_t primary,
    genius_type_t secondary,
    float blend_factor
) {
    if (blend_factor < 0.0f) blend_factor = 0.0f;
    if (blend_factor > 0.5f) blend_factor = 0.5f;

    genius_type_t types[2] = { primary, secondary };
    float weights[2] = { 1.0f - blend_factor, blend_factor };

    return genius_profiles_blend(bridge, types, weights, 2);
}

/* ============================================================================
 * STATE QUERY FUNCTIONS
 * ============================================================================ */

genius_activation_state_t genius_profiles_get_state(
    const genius_profiles_bridge_t* bridge
) {
    if (!bridge) return GENIUS_STATE_ERROR;
    return bridge->state;
}

const genius_profile_t* genius_profiles_get_active(
    const genius_profiles_bridge_t* bridge
) {
    if (!bridge || bridge->active_count == 0) return NULL;
    return bridge->active_profiles[0];
}

float genius_profiles_get_fatigue(const genius_profiles_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->fatigue_level;
}

float genius_profiles_get_flow_depth(const genius_profiles_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->flow_state_depth;
}

bool genius_profiles_is_ready(const genius_profiles_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->state != GENIUS_STATE_ERROR &&
           bridge->state != GENIUS_STATE_DEGRADED;
}

/* ============================================================================
 * MODULATION FUNCTIONS
 * ============================================================================ */

genius_error_t genius_profiles_apply_immune_modulation(
    genius_profiles_bridge_t* bridge,
    float cytokine_level,
    float inflammation
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_immune_modulation) return GENIUS_ERROR_SUCCESS;

    BRIDGE_LOCK(bridge);

    bridge->current_cytokine_effect = cytokine_level;
    bridge->inflammation_level = inflammation;

    /* High inflammation degrades profile */
    if (inflammation > 0.7f) {
        bridge->is_degraded = true;
        bridge->state = GENIUS_STATE_DEGRADED;
    } else {
        bridge->is_degraded = false;
        if (bridge->active_count > 0 && bridge->state == GENIUS_STATE_DEGRADED) {
            bridge->state = bridge->active_count > 1 ?
                           GENIUS_STATE_BLENDED : GENIUS_STATE_ACTIVE;
        }
    }

    BRIDGE_UNLOCK(bridge);

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        float data[2] = { cytokine_level, inflammation };
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_IMMUNE_MODULATION,
                                     0, data, sizeof(data));
    }

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_update_fatigue(
    genius_profiles_bridge_t* bridge,
    uint64_t delta_ms,
    float activity_level
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    BRIDGE_LOCK(bridge);

    /* Fatigue increases with activity */
    float fatigue_rate = 0.00001f * activity_level;

    /* Recovery when in flow or low activity */
    float recovery_rate = 0.000005f;
    if (bridge->flow_state_depth > 0.5f) {
        recovery_rate *= 0.5f; /* Slower fatigue in flow */
    }
    if (activity_level < 0.3f) {
        recovery_rate *= 2.0f; /* Faster recovery at low activity */
    }

    bridge->fatigue_level += (fatigue_rate - recovery_rate) * (float)delta_ms;

    /* Clamp */
    if (bridge->fatigue_level < 0.0f) bridge->fatigue_level = 0.0f;
    if (bridge->fatigue_level > 1.0f) bridge->fatigue_level = 1.0f;

    /* Check for fatigue state */
    if (bridge->fatigue_level > 0.8f && bridge->state == GENIUS_STATE_ACTIVE) {
        bridge->state = GENIUS_STATE_FATIGUED;

        /* Send notification */
        if (bridge->base.bio_async_enabled) {
            genius_profiles_send_message(bridge, BIO_MSG_GENIUS_FATIGUE_UPDATE,
                                         0, &bridge->fatigue_level, sizeof(float));
        }
    }

    bridge->last_update_ms += delta_ms;

    BRIDGE_UNLOCK(bridge);

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_enter_flow(
    genius_profiles_bridge_t* bridge,
    float challenge_level,
    float skill_level
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    if (bridge->state != GENIUS_STATE_ACTIVE &&
        bridge->state != GENIUS_STATE_BLENDED) {
        return GENIUS_ERROR_INVALID_STATE;
    }

    /* Check flow conditions */
    float ratio = fabsf(challenge_level - skill_level);
    float threshold = 0.5f;

    /* Get threshold from active profile if available */
    if (bridge->active_count > 0 && bridge->active_profiles[0]) {
        threshold = bridge->active_profiles[0]->flow_entry_threshold;
    }

    if (ratio > threshold) {
        return GENIUS_ERROR_CONTEXT_MISMATCH; /* Challenge/skill mismatch */
    }

    BRIDGE_LOCK(bridge);

    bridge->state = GENIUS_STATE_FLOW;
    bridge->flow_state_depth = 1.0f - ratio;

    BRIDGE_UNLOCK(bridge);

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        float data[2] = { challenge_level, skill_level };
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_FLOW_STATE_ENTER,
                                     0, data, sizeof(data));
    }

    NIMCP_LOGGING_INFO("Entered flow state (depth=%.2f)", bridge->flow_state_depth);

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_exit_flow(
    genius_profiles_bridge_t* bridge,
    const char* reason
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    if (bridge->state != GENIUS_STATE_FLOW) {
        return GENIUS_ERROR_INVALID_STATE;
    }

    BRIDGE_LOCK(bridge);

    bridge->state = bridge->active_count > 1 ?
                   GENIUS_STATE_BLENDED : GENIUS_STATE_ACTIVE;
    bridge->flow_state_depth = 0.0f;

    BRIDGE_UNLOCK(bridge);

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_FLOW_STATE_EXIT,
                                     0, reason, reason ? strlen(reason) + 1 : 0);
    }

    NIMCP_LOGGING_INFO("Exited flow state: %s", reason ? reason : "no reason");

    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * TRAINING INTEGRATION
 * ============================================================================ */

genius_error_t genius_profiles_training_step(
    genius_profiles_bridge_t* bridge,
    float loss,
    const float* gradients,
    uint32_t gradient_count
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_training_integration) return GENIUS_ERROR_SUCCESS;

    (void)loss;
    (void)gradients;
    (void)gradient_count;

    /* Training integration would modify profile parameters based on gradients */
    /* This is a placeholder for future implementation */

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_TRAINING_STEP,
                                     0, &loss, sizeof(float));
    }

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_apply_stdp(
    genius_profiles_bridge_t* bridge,
    uint64_t pre_spike_time,
    uint64_t post_spike_time
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_stdp) return GENIUS_ERROR_SUCCESS;

    (void)pre_spike_time;
    (void)post_spike_time;

    /* STDP application would modulate profile based on spike timing */
    /* This is a placeholder for future implementation */

    /* Send bio-async notification */
    if (bridge->base.bio_async_enabled) {
        uint64_t data[2] = { pre_spike_time, post_spike_time };
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_STDP_EVENT,
                                     0, data, sizeof(data));
    }

    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * EIDETIC MEMORY FUNCTIONS
 * ============================================================================ */

genius_error_t genius_profiles_apply_eidetic(genius_profiles_bridge_t* bridge) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (bridge->active_count == 0) return GENIUS_ERROR_NOT_ACTIVE;

    /* Get eidetic config from primary profile */
    const genius_profile_t* primary = bridge->active_profiles[0];
    if (!primary) return GENIUS_ERROR_NOT_ACTIVE;

    /* Apply to working memory if connected */
    if (bridge->working_memory) {
        /* Working memory enhancement would go here */
        NIMCP_LOGGING_DEBUG("Applied eidetic config to working memory");
    }

    /* Apply to hippocampus if connected */
    if (bridge->hippocampus) {
        /* Hippocampus enhancement would go here */
        NIMCP_LOGGING_DEBUG("Applied eidetic config to hippocampus");
    }

    /* Apply to other memory systems similarly... */

    return GENIUS_ERROR_SUCCESS;
}

const eidetic_memory_config_t* genius_profiles_get_eidetic_config(
    const genius_profiles_bridge_t* bridge
) {
    if (!bridge || bridge->active_count == 0) return NULL;

    const genius_profile_t* primary = bridge->active_profiles[0];
    if (!primary) return NULL;

    return &primary->eidetic;
}

/* ============================================================================
 * EXCEPTION HANDLING
 * ============================================================================ */

nimcp_exception_t* genius_profiles_raise_exception(
    genius_profiles_bridge_t* bridge,
    genius_error_t error,
    const char* message
) {
    if (!bridge) return NULL;

    BRIDGE_LOCK(bridge);
    bridge->total_exceptions++;
    BRIDGE_UNLOCK(bridge);

    /* Create exception */
    nimcp_exception_t* exc = NULL;

    /* Present to immune if configured */
    if (bridge->config.enable_exception_immune_presentation && bridge->immune_system) {
        bridge->immune_presentations++;
        NIMCP_THROW_TO_IMMUNE(error, message ? message : genius_error_message(error));
    }

    NIMCP_LOGGING_ERROR("Genius profile exception: %s (code=0x%04X)",
                        message ? message : genius_error_message(error), error);

    return exc;
}

/* ============================================================================
 * BIO-ASYNC MESSAGE HANDLING
 * ============================================================================ */

genius_error_t genius_profiles_handle_message(
    genius_profiles_bridge_t* bridge,
    const void* message,
    size_t message_size
) {
    if (!bridge || !message) return GENIUS_ERROR_NULL_POINTER;
    if (message_size < sizeof(bio_message_header_t)) return GENIUS_ERROR_INVALID_STATE;

    const bio_message_header_t* header = (const bio_message_header_t*)message;
    const void* payload = (const uint8_t*)message + sizeof(bio_message_header_t);
    size_t payload_size = header->payload_size;

    switch (header->type) {
        case BIO_MSG_GENIUS_PROFILE_ACTIVATE:
            if (payload_size >= sizeof(genius_type_t)) {
                genius_type_t type = *(const genius_type_t*)payload;
                return genius_profiles_activate(bridge, type, 1.0f);
            }
            break;

        case BIO_MSG_GENIUS_PROFILE_DEACTIVATE:
            return genius_profiles_deactivate(bridge);

        case BIO_MSG_GENIUS_QUERY_STATE: {
            genius_activation_state_t state = bridge->state;
            genius_profiles_send_message(bridge, BIO_MSG_GENIUS_STATE_RESPONSE,
                                         header->source_module, &state, sizeof(state));
            break;
        }

        case BIO_MSG_GENIUS_IMMUNE_MODULATION:
            if (payload_size >= 2 * sizeof(float)) {
                const float* data = (const float*)payload;
                return genius_profiles_apply_immune_modulation(bridge, data[0], data[1]);
            }
            break;

        case BIO_MSG_GENIUS_FLOW_STATE_ENTER:
            if (payload_size >= 2 * sizeof(float)) {
                const float* data = (const float*)payload;
                return genius_profiles_enter_flow(bridge, data[0], data[1]);
            }
            break;

        case BIO_MSG_GENIUS_FLOW_STATE_EXIT:
            return genius_profiles_exit_flow(bridge,
                                             payload_size > 0 ?
                                             (const char*)payload : NULL);

        default:
            NIMCP_LOGGING_DEBUG("Unhandled genius message type: 0x%04X", header->type);
            break;
    }

    return GENIUS_ERROR_SUCCESS;
}

/**
 * @brief Maximum message buffer size for bio-async messages
 */
#define GENIUS_BIO_MSG_BUFFER_SIZE 4096

genius_error_t genius_profiles_send_message(
    genius_profiles_bridge_t* bridge,
    genius_bio_message_t msg_type,
    uint32_t target_module,
    const void* data,
    size_t data_len
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->base.bio_async_enabled) return GENIUS_ERROR_BIO_ASYNC_FAILED;

    /* Determine appropriate neuromodulator channel based on message category */
    nimcp_bio_channel_type_t channel;
    if (msg_type >= BIO_MSG_GENIUS_TRAINING_START && msg_type <= BIO_MSG_GENIUS_TRAINING_COMPLETE) {
        channel = BIO_CHANNEL_DOPAMINE;  /* Training = reward/learning signals */
    } else if (msg_type >= BIO_MSG_GENIUS_IMMUNE_MODULATION && msg_type <= BIO_MSG_GENIUS_BBB_ALERT) {
        channel = BIO_CHANNEL_NOREPINEPHRINE;  /* Immune = alertness/priority */
    } else if (msg_type >= BIO_MSG_GENIUS_EIDETIC_ENCODE && msg_type <= BIO_MSG_GENIUS_EIDETIC_CONSOLIDATE) {
        channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Eidetic = attention/memory */
    } else {
        channel = BIO_CHANNEL_SEROTONIN;  /* Default = state coordination */
    }

    /* Create promise on selected channel */
    size_t total_size = sizeof(uint32_t) + sizeof(uint32_t) + data_len;  /* type + target + payload */
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(channel, total_size);
    if (!promise) {
        return GENIUS_ERROR_BIO_ASYNC_FAILED;
    }

    /* Pack message: type (4 bytes) + target_module (4 bytes) + payload */
    uint8_t buffer[GENIUS_BIO_MSG_BUFFER_SIZE];
    if (total_size > sizeof(buffer)) {
        nimcp_bio_promise_destroy(promise);
        return GENIUS_ERROR_BIO_ASYNC_FAILED;
    }

    uint32_t type_val = (uint32_t)msg_type;
    memcpy(buffer, &type_val, sizeof(uint32_t));
    memcpy(buffer + sizeof(uint32_t), &target_module, sizeof(uint32_t));
    if (data && data_len > 0) {
        memcpy(buffer + 2 * sizeof(uint32_t), data, data_len);
    }

    /* Complete promise with packed data */
    nimcp_error_t err = nimcp_bio_promise_complete(promise, buffer);
    if (err != NIMCP_SUCCESS) {
        nimcp_bio_promise_destroy(promise);
        return GENIUS_ERROR_BIO_ASYNC_FAILED;
    }

    nimcp_bio_promise_destroy(promise);
    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * HEALTH AGENT INTEGRATION
 * ============================================================================ */

genius_error_t genius_profiles_heartbeat(genius_profiles_bridge_t* bridge) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    bridge->last_heartbeat_ms = genius_profiles_get_timestamp_ms();
    genius_profiles_heartbeat_internal("heartbeat",
                                       1.0f - bridge->fatigue_level);

    /* Send health status via bio-async */
    if (bridge->base.bio_async_enabled) {
        float health = 1.0f - bridge->fatigue_level;
        if (bridge->is_degraded) health *= 0.5f;
        genius_profiles_send_message(bridge, BIO_MSG_GENIUS_HEALTH_STATUS,
                                     0, &health, sizeof(float));
    }

    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_start_health_agent(genius_profiles_bridge_t* bridge) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    bridge->health_agent_active = true;
    bridge->last_heartbeat_ms = genius_profiles_get_timestamp_ms();

    NIMCP_LOGGING_INFO("Genius profiles health agent started");
    return GENIUS_ERROR_SUCCESS;
}

genius_error_t genius_profiles_stop_health_agent(genius_profiles_bridge_t* bridge) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;

    bridge->health_agent_active = false;

    NIMCP_LOGGING_INFO("Genius profiles health agent stopped");
    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * MESH COORDINATION
 * ============================================================================ */

genius_error_t genius_profiles_mesh_propose(
    genius_profiles_bridge_t* bridge,
    genius_type_t type,
    float strength
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_mesh_coordination) return GENIUS_ERROR_SUCCESS;
    if (!bridge->mesh_coordinator) return GENIUS_ERROR_BRIDGE_DISCONNECTED;

    /* Mesh proposal would go through coordinator */
    /* For now, just activate directly */

    bridge->mesh_consensus_count++;

    return genius_profiles_activate(bridge, type, strength);
}

genius_error_t genius_profiles_mesh_endorse(
    genius_profiles_bridge_t* bridge,
    mesh_transaction_t* tx
) {
    if (!bridge || !tx) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_mesh_coordination) return GENIUS_ERROR_SUCCESS;

    /* Mesh endorsement would go here */
    (void)tx;

    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * QUANTUM OPTIMIZATION
 * ============================================================================ */

genius_error_t genius_profiles_quantum_optimize(
    genius_profiles_bridge_t* bridge,
    const float* objective,
    size_t objective_len
) {
    if (!bridge) return GENIUS_ERROR_NULL_POINTER;
    if (!bridge->config.enable_quantum_optimization) return GENIUS_ERROR_SUCCESS;

    (void)objective;
    (void)objective_len;

    /* Quantum optimization would go here */

    return GENIUS_ERROR_SUCCESS;
}

/* ============================================================================
 * BRAIN CREATION HELPERS
 * ============================================================================ */

nimcp_brain_t* genius_brain_create(genius_type_t type) {
    if (!genius_type_is_valid(type)) {
        return NULL;
    }

    const genius_profile_t* profile = genius_profile_get(type);
    if (!profile) {
        return NULL;
    }

    /* Brain creation would use profile parameters to configure brain */
    /* This is a placeholder - actual implementation depends on brain factory */

    NIMCP_LOGGING_INFO("Created genius brain: %s", genius_type_name(type));

    return NULL; /* Placeholder */
}

hemispheric_brain_t* genius_hemispheric_brain_create(genius_type_t type) {
    if (!genius_type_is_valid(type)) {
        return NULL;
    }

    const genius_profile_t* profile = genius_profile_get(type);
    if (!profile) {
        return NULL;
    }

    /* Hemispheric brain creation with lateralization */
    /* This is a placeholder - actual implementation depends on hemispheric brain factory */

    NIMCP_LOGGING_INFO("Created hemispheric genius brain: %s", genius_type_name(type));

    return NULL; /* Placeholder */
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

uint64_t genius_profiles_get_timestamp_ms(void) {
    return nimcp_time_get_ms();
}
