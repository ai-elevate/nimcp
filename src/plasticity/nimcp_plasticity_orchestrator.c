/**
 * @file nimcp_plasticity_orchestrator.c
 * @brief Unified Plasticity Orchestrator - Full Implementation
 * @version 1.0.0
 * @date 2025-12-19
 *
 * WHAT: Central coordinator for all NIMCP plasticity mechanisms
 * WHY:  Enables coherent multi-mechanism plasticity with proper sequencing
 * HOW:  Orchestrates STDP, BCM, homeostatic, structural, metabolic, etc.
 *
 * Orchestration Order (biologically-motivated):
 * 1. Metabolic check - ensure sufficient ATP
 * 2. Calcium dynamics - compute [Ca²⁺] from spike activity
 * 3. STDP/Triplet STDP - Hebbian weight changes
 * 4. Heterosynaptic - neighbor depression
 * 5. BCM - threshold-based selectivity
 * 6. Homeostatic - maintain target rates
 * 7. Metaplasticity - update sliding thresholds
 * 8. Protein synthesis - tag capture for consolidation
 * 9. Structural - spine dynamics
 * 10. Astrocyte - glial modulation
 */

#include "plasticity/nimcp_plasticity_orchestrator.h"
#include "plasticity/stdp/nimcp_triplet_stdp.h"
#include "plasticity/bcm/nimcp_bcm.h"
#include "plasticity/homeostatic/nimcp_homeostatic.h"
#include "plasticity/calcium/nimcp_calcium_dynamics.h"
#include "plasticity/structural/nimcp_structural_plasticity.h"
#include "plasticity/metabolic/nimcp_metabolic_plasticity.h"
#include "plasticity/heterosynaptic/nimcp_heterosynaptic.h"
#include "plasticity/astrocyte/nimcp_astrocyte_plasticity.h"
#include "plasticity/protein/nimcp_protein_synthesis.h"
#include "plasticity/metaplasticity/nimcp_extended_metaplasticity.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/validation/nimcp_common.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "plasticity_orchestrator"
#define MAX_SYNAPSES 100
#define MAX_NEURONS 50
#define MAX_CALLBACKS 32
#define MAX_ASTROCYTES 100

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

typedef struct {
    plasticity_event_callback_t callback;
    void* user_data;
    plasticity_event_type_t event_type;
    int id;
    bool active;
} event_callback_entry_t;

typedef struct {
    plasticity_update_callback_t callback;
    void* user_data;
    bool active;
} update_callback_entry_t;

/**
 * @brief Synapse entry in orchestrator
 *
 * Contains all per-synapse plasticity module states
 */
typedef struct {
    uint32_t id;
    uint32_t pre_neuron_id;
    uint32_t post_neuron_id;
    float weight;
    float w_min;
    float w_max;

    /* Module-specific states */
    triplet_stdp_synapse_t* triplet_stdp;
    bcm_synapse_t bcm_state;
    uint32_t structural_synapse_id;  /* ID in structural plasticity system */

    /* Activity tracking */
    uint64_t last_pre_spike_time;
    uint64_t last_post_spike_time;
    uint64_t total_pre_spikes;
    uint64_t total_post_spikes;
    float recent_activity_hz;
    float eligibility_trace;

    /* Consolidation state */
    bool consolidation_tagged;
    float ltp_accumulator;
    float ltd_accumulator;

    bool active;
} synapse_entry_t;

/**
 * @brief Neuron entry in orchestrator
 */
typedef struct {
    uint32_t id;
    float firing_rate;
    float bcm_threshold;
    float target_rate;

    /* Homeostatic state */
    synaptic_scaling_state_t scaling_state;
    intrinsic_plasticity_state_t ip_state;
    metaplasticity_state_t meta_state;

    /* Associated synapses (indices into synapse array) */
    uint32_t* input_synapse_indices;
    uint32_t num_inputs;
    uint32_t input_capacity;

    bool active;
} neuron_entry_t;

/**
 * @brief Internal orchestrator state
 */
struct plasticity_orchestrator_struct {
    /* Configuration */
    plasticity_orchestrator_config_t config;

    /* Synapse and neuron arrays */
    synapse_entry_t* synapses;
    neuron_entry_t* neurons;
    size_t num_synapses;
    size_t num_neurons;
    size_t synapse_capacity;
    size_t neuron_capacity;

    /* Global plasticity modules */
    metabolic_plasticity_t* metabolic;
    calcium_dynamics_t calcium;
    structural_plasticity_system_t* structural;
    hetero_system_t* heterosynaptic;
    astrocyte_plasticity_t astrocyte;
    protein_synthesis_system_t protein_synthesis;
    homeostatic_controller_t homeostatic;
    metaplasticity_controller_t metaplasticity;

    /* Integration handles */
    struct brain_immune_system* immune;
    struct sleep_system_struct* sleep;
    struct neuromodulator_system_struct* neuromod;

    /* Current modulation state */
    sleep_state_t current_sleep_state;
    float sleep_modulation_factor;
    float immune_modulation_factor;
    neuromodulator_levels_t neuromod_levels;

    /* Callbacks */
    event_callback_entry_t event_callbacks[MAX_CALLBACKS];
    update_callback_entry_t pre_update_callbacks[MAX_CALLBACKS];
    update_callback_entry_t post_update_callbacks[MAX_CALLBACKS];
    int next_callback_id;

    /* Statistics */
    plasticity_stats_t stats;

    /* State */
    uint64_t current_time_ms;
    uint64_t last_consolidation_time;
    uint64_t last_homeostatic_time;
    uint64_t last_structural_time;

    /* Bio-async */
    bool bio_async_connected;

    /* Thread safety */
    nimcp_platform_mutex_t* mutex;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static void emit_event(
    plasticity_orchestrator_t* orchestrator,
    plasticity_event_type_t type,
    uint32_t synapse_id,
    uint32_t neuron_id,
    float old_val,
    float new_val
) {
    if (!orchestrator) return;

    plasticity_event_t event = {
        .type = type,
        .synapse_id = synapse_id,
        .neuron_id = neuron_id,
        .old_value = old_val,
        .new_value = new_val,
        .delta = new_val - old_val,
        .timestamp_ms = orchestrator->current_time_ms,
        .context = NULL
    };

    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (orchestrator->event_callbacks[i].active &&
            orchestrator->event_callbacks[i].event_type == type) {
            orchestrator->event_callbacks[i].callback(
                &event,
                orchestrator->event_callbacks[i].user_data
            );
        }
    }

    /* Update stats */
    switch (type) {
        case PLASTICITY_EVENT_LTP:
            orchestrator->stats.ltp_count++;
            break;
        case PLASTICITY_EVENT_LTD:
            orchestrator->stats.ltd_count++;
            break;
        case PLASTICITY_EVENT_SPINE_FORMED:
            orchestrator->stats.spines_formed++;
            break;
        case PLASTICITY_EVENT_SPINE_ELIMINATED:
            orchestrator->stats.spines_eliminated++;
            break;
        case PLASTICITY_EVENT_CONSOLIDATION:
            orchestrator->stats.consolidation_events++;
            break;
        default:
            break;
    }
}

static synapse_entry_t* find_synapse(plasticity_orchestrator_t* orch, uint32_t id) {
    if (!orch || !orch->synapses) return NULL;

    for (size_t i = 0; i < orch->num_synapses; i++) {
        if (orch->synapses[i].id == id && orch->synapses[i].active) {
            return &orch->synapses[i];
        }
    }
    return NULL;
}

static neuron_entry_t* find_neuron(plasticity_orchestrator_t* orch, uint32_t id) {
    if (!orch || !orch->neurons) return NULL;

    for (size_t i = 0; i < orch->num_neurons; i++) {
        if (orch->neurons[i].id == id && orch->neurons[i].active) {
            return &orch->neurons[i];
        }
    }
    return NULL;
}

/**
 * @brief Structural plasticity change callback
 *
 * WHAT: Handles spine formation/elimination events from structural module
 * WHY:  Converts structural events to orchestrator events for tracking
 * HOW:  Maps structural event types to plasticity events
 */
static void structural_change_handler(
    structural_event_t event,
    uint32_t synapse_id,
    synapse_state_t old_state,
    synapse_state_t new_state,
    void* user_data
) {
    plasticity_orchestrator_t* orch = (plasticity_orchestrator_t*)user_data;
    if (!orch) {
        NIMCP_LOGGING_DEBUG("structural_change_handler: NULL orchestrator");
        return;
    }

    switch (event) {
        case STRUCTURAL_EVENT_FORMATION:
            emit_event(orch, PLASTICITY_EVENT_SPINE_FORMED, synapse_id, 0, 0.0f, 1.0f);
            break;
        case STRUCTURAL_EVENT_ELIMINATION:
            emit_event(orch, PLASTICITY_EVENT_SPINE_ELIMINATED, synapse_id, 0, 1.0f, 0.0f);
            break;
        default:
            /* Other structural events (stabilization, potentiation, etc.) */
            break;
    }
}

static synapse_entry_t* get_or_create_synapse(plasticity_orchestrator_t* orch, uint32_t id) {
    synapse_entry_t* syn = find_synapse(orch, id);
    if (syn) return syn;

    if (orch->num_synapses >= orch->synapse_capacity) {
        NIMCP_LOGGING_WARN("Synapse capacity reached");
        return NULL;
    }

    syn = &orch->synapses[orch->num_synapses++];
    memset(syn, 0, sizeof(synapse_entry_t));
    syn->id = id;
    syn->weight = 0.5f;
    syn->w_min = 0.0f;
    syn->w_max = 1.0f;
    syn->active = true;

    /* Initialize BCM state */
    syn->bcm_state = bcm_synapse_init(syn->weight, 0.5f);

    /* Create triplet STDP synapse if enabled */
    if (orch->config.enabled.enable_triplet_stdp) {
        syn->triplet_stdp = triplet_stdp_synapse_create(NULL, syn->weight);
    }

    /* Register with structural plasticity if enabled */
    if (orch->config.enabled.enable_structural && orch->structural) {
        uint32_t struct_id = 0;
        /* Form synapse with moderate initial activity to trigger formation event */
        if (structural_plasticity_form_synapse(orch->structural, id, id, 25.0f, &struct_id) == 0) {
            syn->structural_synapse_id = struct_id;
        }
    }

    return syn;
}

static neuron_entry_t* get_or_create_neuron(plasticity_orchestrator_t* orch, uint32_t id) {
    neuron_entry_t* neuron = find_neuron(orch, id);
    if (neuron) return neuron;

    if (orch->num_neurons >= orch->neuron_capacity) {
        NIMCP_LOGGING_WARN("Neuron capacity reached");
        return NULL;
    }

    neuron = &orch->neurons[orch->num_neurons++];
    memset(neuron, 0, sizeof(neuron_entry_t));
    neuron->id = id;
    neuron->firing_rate = 0.0f;
    neuron->bcm_threshold = 0.5f;
    neuron->target_rate = 5.0f;  /* Default 5 Hz target */
    neuron->active = true;

    /* Initialize homeostatic states */
    neuron->scaling_state = synaptic_scaling_state_init(neuron->target_rate);
    neuron->ip_state = intrinsic_plasticity_state_init(0.5f, 1.0f);
    neuron->meta_state = metaplasticity_state_init(0.5f);

    return neuron;
}

static float get_sleep_modulation(sleep_state_t state) {
    switch (state) {
        case SLEEP_STATE_AWAKE:
            return 1.0f;
        case SLEEP_STATE_DROWSY:
            return 0.9f;
        case SLEEP_STATE_LIGHT_NREM:
            return 0.7f;
        case SLEEP_STATE_DEEP_NREM:
            return 0.3f;  /* Reduced learning during deep sleep */
        case SLEEP_STATE_REM:
            return 0.5f;  /* Memory consolidation mode */
        default:
            return 1.0f;
    }
}

/**
 * @brief Get immune modulation factor based on inflammation level
 *
 * WHAT: Converts inflammation level to plasticity modulation factor
 * WHY:  Higher inflammation reduces synaptic plasticity (fever model)
 * HOW:  Maps inflammation levels to [0.1, 1.0] range
 */
static float get_immune_modulation(brain_inflammation_level_t level) {
    switch (level) {
        case INFLAMMATION_NONE:
            return 1.0f;   /* Full plasticity */
        case INFLAMMATION_LOCAL:
            return 0.85f;  /* Slight reduction */
        case INFLAMMATION_REGIONAL:
            return 0.65f;  /* Moderate reduction */
        case INFLAMMATION_SYSTEMIC:
            return 0.40f;  /* Severe reduction */
        case INFLAMMATION_STORM:
            return 0.10f;  /* Emergency - minimal plasticity */
        default:
            return 1.0f;
    }
}

static void update_weight_statistics(plasticity_orchestrator_t* orch) {
    if (!orch || orch->num_synapses == 0) return;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    float min_w = 1.0f;
    float max_w = 0.0f;

    for (size_t i = 0; i < orch->num_synapses; i++) {
        if (!orch->synapses[i].active) continue;
        float w = orch->synapses[i].weight;
        sum += w;
        sum_sq += w * w;
        if (w < min_w) min_w = w;
        if (w > max_w) max_w = w;
    }

    size_t n = orch->num_synapses;
    orch->stats.mean_weight = sum / (float)n;
    orch->stats.std_weight = sqrtf((sum_sq / (float)n) -
                                    (orch->stats.mean_weight * orch->stats.mean_weight));
    orch->stats.min_weight = min_w;
    orch->stats.max_weight = max_w;
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

int plasticity_orchestrator_default_config(plasticity_orchestrator_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL config in default_config");
        NIMCP_LOGGING_ERROR("NULL config in default_config");
        return -1;
    }

    memset(config, 0, sizeof(plasticity_orchestrator_config_t));

    /* Enable all mechanisms by default */
    config->enabled.enable_classic_stdp = false;  /* Use triplet STDP instead */
    config->enabled.enable_triplet_stdp = true;
    config->enabled.enable_bcm = true;
    config->enabled.enable_homeostatic = true;
    config->enabled.enable_eligibility = true;
    config->enabled.enable_stp = false;  /* Optional */
    config->enabled.enable_dendritic = false;  /* Optional */
    config->enabled.enable_predictive = false;  /* Optional */
    config->enabled.enable_structural = true;
    config->enabled.enable_heterosynaptic = true;
    config->enabled.enable_calcium = true;
    config->enabled.enable_astrocyte = true;
    config->enabled.enable_protein_synthesis = true;
    config->enabled.enable_metaplasticity = true;
    config->enabled.enable_metabolic = true;

    /* Global modulation */
    config->global_learning_rate = 1.0f;
    config->sleep_modulation = 1.0f;
    config->immune_modulation = 1.0f;

    /* Timing */
    config->update_interval_ms = 1;
    config->consolidation_interval_ms = 60000;  /* 1 minute */
    config->homeostatic_interval_ms = 1000;     /* 1 second */

    /* Integration */
    config->connect_sleep_bridges = true;
    config->connect_immune_bridges = true;
    config->connect_bio_async = false;

    config->log_level = 2;  /* Warnings */

    return 0;
}

plasticity_orchestrator_t* plasticity_orchestrator_create(
    const plasticity_orchestrator_config_t* config
) {
    plasticity_orchestrator_config_t default_config;
    if (!config) {
        plasticity_orchestrator_default_config(&default_config);
        config = &default_config;
    }

    plasticity_orchestrator_t* orchestrator = (plasticity_orchestrator_t*)nimcp_malloc(
        sizeof(plasticity_orchestrator_t)
    );

    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate plasticity orchestrator");
        NIMCP_LOGGING_ERROR("Failed to allocate plasticity orchestrator");
        return NULL;
    }

    memset(orchestrator, 0, sizeof(plasticity_orchestrator_t));
    orchestrator->config = *config;
    orchestrator->next_callback_id = 1;

    /* Create mutex */
    orchestrator->mutex = nimcp_platform_mutex_create();
    if (!orchestrator->mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to create mutex");
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(orchestrator);
        return NULL;
    }

    /* Allocate synapse and neuron arrays */
    orchestrator->synapse_capacity = MAX_SYNAPSES;
    orchestrator->neuron_capacity = MAX_NEURONS;

    orchestrator->synapses = (synapse_entry_t*)nimcp_malloc(
        sizeof(synapse_entry_t) * MAX_SYNAPSES
    );
    orchestrator->neurons = (neuron_entry_t*)nimcp_malloc(
        sizeof(neuron_entry_t) * MAX_NEURONS
    );

    if (!orchestrator->synapses || !orchestrator->neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate synapse/neuron arrays");
        NIMCP_LOGGING_ERROR("Failed to allocate synapse/neuron arrays");
        if (orchestrator->synapses) nimcp_free(orchestrator->synapses);
        if (orchestrator->neurons) nimcp_free(orchestrator->neurons);
        nimcp_platform_mutex_destroy(orchestrator->mutex);
        nimcp_free(orchestrator);
        return NULL;
    }

    memset(orchestrator->synapses, 0, sizeof(synapse_entry_t) * MAX_SYNAPSES);
    memset(orchestrator->neurons, 0, sizeof(neuron_entry_t) * MAX_NEURONS);

    /* Initialize metabolic system */
    if (config->enabled.enable_metabolic) {
        metabolic_config_t met_config;
        metabolic_plasticity_default_config(&met_config);
        orchestrator->metabolic = metabolic_plasticity_create(&met_config);
    }

    /* Initialize calcium dynamics */
    if (config->enabled.enable_calcium) {
        calcium_config_t ca_config;
        calcium_default_config(&ca_config);
        orchestrator->calcium = calcium_create(&ca_config);
    }

    /* Initialize structural plasticity */
    if (config->enabled.enable_structural) {
        structural_plasticity_config_t struct_config;
        structural_plasticity_default_config(&struct_config);
        orchestrator->structural = structural_plasticity_create(&struct_config);
        /* Register callback for spine formation/elimination events */
        if (orchestrator->structural) {
            structural_plasticity_register_callback(
                orchestrator->structural, structural_change_handler, orchestrator);
        }
    }

    /* Initialize heterosynaptic system */
    if (config->enabled.enable_heterosynaptic) {
        hetero_config_t hetero_config;
        hetero_default_config(&hetero_config);
        orchestrator->heterosynaptic = hetero_create(&hetero_config, 1000);
    }

    /* Initialize astrocyte system */
    if (config->enabled.enable_astrocyte) {
        astrocyte_config_t astro_config;
        astrocyte_plasticity_default_config(&astro_config);
        orchestrator->astrocyte = astrocyte_plasticity_create(&astro_config, MAX_ASTROCYTES);
    }

    /* Initialize protein synthesis */
    if (config->enabled.enable_protein_synthesis) {
        protein_synthesis_config_t prot_config;
        protein_synthesis_default_config(&prot_config);
        orchestrator->protein_synthesis = protein_synthesis_create(&prot_config);
    }

    /* Initialize homeostatic controller */
    if (config->enabled.enable_homeostatic) {
        homeostatic_config_t homeo_config = homeostatic_config_default();
        orchestrator->homeostatic = homeostatic_controller_create(&homeo_config, MAX_NEURONS);
    }

    /* Initialize metaplasticity controller */
    if (config->enabled.enable_metaplasticity) {
        extended_metaplasticity_config_t meta_config = metaplasticity_config_default();
        orchestrator->metaplasticity = metaplasticity_controller_create(&meta_config, MAX_SYNAPSES);
    }

    /* Initialize default modulation */
    orchestrator->current_sleep_state = SLEEP_STATE_AWAKE;
    orchestrator->sleep_modulation_factor = 1.0f;
    orchestrator->immune_modulation_factor = 1.0f;
    orchestrator->neuromod_levels.dopamine = 0.5f;
    orchestrator->neuromod_levels.norepinephrine = 0.3f;
    orchestrator->neuromod_levels.acetylcholine = 0.5f;
    orchestrator->neuromod_levels.serotonin = 0.4f;

    /* Initialize statistics */
    orchestrator->stats.mean_weight = 0.5f;
    orchestrator->stats.mean_atp = 100.0f;
    orchestrator->stats.min_atp = 100.0f;

    NIMCP_LOGGING_INFO("Plasticity orchestrator created with all modules");
    return orchestrator;
}

void plasticity_orchestrator_destroy(plasticity_orchestrator_t* orchestrator) {
    if (!orchestrator) return;

    nimcp_platform_mutex_lock(orchestrator->mutex);

    /* Destroy synapse module states */
    for (size_t i = 0; i < orchestrator->num_synapses; i++) {
        if (orchestrator->synapses[i].triplet_stdp) {
            triplet_stdp_synapse_destroy(orchestrator->synapses[i].triplet_stdp);
        }
    }

    /* Destroy neuron input arrays */
    for (size_t i = 0; i < orchestrator->num_neurons; i++) {
        if (orchestrator->neurons[i].input_synapse_indices) {
            nimcp_free(orchestrator->neurons[i].input_synapse_indices);
        }
    }

    /* Destroy global modules */
    if (orchestrator->metabolic) {
        metabolic_plasticity_destroy(orchestrator->metabolic);
    }
    if (orchestrator->calcium) {
        calcium_destroy(orchestrator->calcium);
    }
    if (orchestrator->structural) {
        structural_plasticity_destroy(orchestrator->structural);
    }
    if (orchestrator->heterosynaptic) {
        hetero_destroy(orchestrator->heterosynaptic);
    }
    if (orchestrator->astrocyte) {
        astrocyte_plasticity_destroy(orchestrator->astrocyte);
    }
    if (orchestrator->protein_synthesis) {
        protein_synthesis_destroy(orchestrator->protein_synthesis);
    }
    if (orchestrator->homeostatic) {
        homeostatic_controller_destroy(orchestrator->homeostatic);
    }
    if (orchestrator->metaplasticity) {
        metaplasticity_controller_destroy(orchestrator->metaplasticity);
    }

    /* Free arrays */
    if (orchestrator->synapses) nimcp_free(orchestrator->synapses);
    if (orchestrator->neurons) nimcp_free(orchestrator->neurons);

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    nimcp_platform_mutex_destroy(orchestrator->mutex);

    nimcp_free(orchestrator);

    NIMCP_LOGGING_DEBUG("Destroyed plasticity orchestrator");
}

/* ============================================================================
 * Integration Functions
 * ============================================================================ */

int plasticity_orchestrator_connect_immune(
    plasticity_orchestrator_t* orchestrator,
    struct brain_immune_system* immune
) {
    if (!orchestrator || !immune) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in connect_immune");
        NIMCP_LOGGING_ERROR("NULL pointer in connect_immune");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->immune = immune;
    nimcp_platform_mutex_unlock(orchestrator->mutex);

    NIMCP_LOGGING_INFO("Connected orchestrator to immune system");
    return 0;
}

int plasticity_orchestrator_connect_sleep(
    plasticity_orchestrator_t* orchestrator,
    struct sleep_system_struct* sleep
) {
    if (!orchestrator || !sleep) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in connect_sleep");
        NIMCP_LOGGING_ERROR("NULL pointer in connect_sleep");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->sleep = sleep;
    nimcp_platform_mutex_unlock(orchestrator->mutex);

    NIMCP_LOGGING_INFO("Connected orchestrator to sleep system");
    return 0;
}

int plasticity_orchestrator_connect_neuromodulators(
    plasticity_orchestrator_t* orchestrator,
    struct neuromodulator_system_struct* neuromod
) {
    if (!orchestrator || !neuromod) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in connect_neuromodulators");
        NIMCP_LOGGING_ERROR("NULL pointer in connect_neuromodulators");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->neuromod = neuromod;
    nimcp_platform_mutex_unlock(orchestrator->mutex);

    NIMCP_LOGGING_INFO("Connected orchestrator to neuromodulator system");
    return 0;
}

int plasticity_orchestrator_connect_bio_async(plasticity_orchestrator_t* orchestrator) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in connect_bio_async");
        return -1;
    }

    orchestrator->bio_async_connected = true;
    NIMCP_LOGGING_INFO("Connected orchestrator to bio-async router");
    return 0;
}

/* ============================================================================
 * Callback Registration
 * ============================================================================ */

int plasticity_orchestrator_register_event_callback(
    plasticity_orchestrator_t* orchestrator,
    plasticity_event_type_t event_type,
    plasticity_event_callback_t callback,
    void* user_data
) {
    if (!orchestrator || !callback) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in register_event_callback");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (!orchestrator->event_callbacks[i].active) {
            orchestrator->event_callbacks[i].callback = callback;
            orchestrator->event_callbacks[i].user_data = user_data;
            orchestrator->event_callbacks[i].event_type = event_type;
            orchestrator->event_callbacks[i].id = orchestrator->next_callback_id++;
            orchestrator->event_callbacks[i].active = true;

            int id = orchestrator->event_callbacks[i].id;
            nimcp_platform_mutex_unlock(orchestrator->mutex);
            return id;
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return -1;
}

int plasticity_orchestrator_unregister_event_callback(
    plasticity_orchestrator_t* orchestrator,
    int callback_id
) {
    if (!orchestrator || callback_id <= 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Invalid parameters in unregister_event_callback");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (orchestrator->event_callbacks[i].active &&
            orchestrator->event_callbacks[i].id == callback_id) {
            orchestrator->event_callbacks[i].active = false;
            nimcp_platform_mutex_unlock(orchestrator->mutex);
            return 0;
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return -1;
}

int plasticity_orchestrator_register_pre_update(
    plasticity_orchestrator_t* orchestrator,
    plasticity_update_callback_t callback,
    void* user_data
) {
    if (!orchestrator || !callback) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in register_pre_update");
        return -1;
    }

    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (!orchestrator->pre_update_callbacks[i].active) {
            orchestrator->pre_update_callbacks[i].callback = callback;
            orchestrator->pre_update_callbacks[i].user_data = user_data;
            orchestrator->pre_update_callbacks[i].active = true;
            return 0;
        }
    }
    return -1;
}

int plasticity_orchestrator_register_post_update(
    plasticity_orchestrator_t* orchestrator,
    plasticity_update_callback_t callback,
    void* user_data
) {
    if (!orchestrator || !callback) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in register_post_update");
        return -1;
    }

    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (!orchestrator->post_update_callbacks[i].active) {
            orchestrator->post_update_callbacks[i].callback = callback;
            orchestrator->post_update_callbacks[i].user_data = user_data;
            orchestrator->post_update_callbacks[i].active = true;
            return 0;
        }
    }
    return -1;
}

/* ============================================================================
 * Main Update Function
 * ============================================================================ */

int plasticity_orchestrator_update(
    plasticity_orchestrator_t* orchestrator,
    uint64_t delta_ms
) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in update");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    orchestrator->current_time_ms += delta_ms;
    float dt = (float)delta_ms;

    /* Pre-update callbacks */
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (orchestrator->pre_update_callbacks[i].active) {
            orchestrator->pre_update_callbacks[i].callback(
                orchestrator,
                delta_ms,
                orchestrator->pre_update_callbacks[i].user_data
            );
        }
    }

    /* Update sleep modulation */
    orchestrator->sleep_modulation_factor = get_sleep_modulation(orchestrator->current_sleep_state);

    /* Update immune modulation */
    if (orchestrator->immune) {
        brain_inflammation_level_t inflammation = brain_immune_get_inflammation_level(
            (brain_immune_system_t*)orchestrator->immune);
        orchestrator->immune_modulation_factor = get_immune_modulation(inflammation);
    }

    /* Compute effective learning rate */
    float effective_lr = orchestrator->config.global_learning_rate *
                         orchestrator->sleep_modulation_factor *
                         orchestrator->immune_modulation_factor;

    /* ============ ORCHESTRATION ORDER ============ */

    /* 1. Metabolic check - ensure sufficient ATP */
    if (orchestrator->config.enabled.enable_metabolic && orchestrator->metabolic) {
        metabolic_plasticity_update(orchestrator->metabolic, delta_ms);

        float atp = metabolic_plasticity_get_atp_level(orchestrator->metabolic);
        orchestrator->stats.mean_atp = atp;
        if (atp < orchestrator->stats.min_atp) {
            orchestrator->stats.min_atp = atp;
        }

        /* Check if plasticity is blocked */
        if (!metabolic_plasticity_can_ltp(orchestrator->metabolic) &&
            !metabolic_plasticity_can_ltd(orchestrator->metabolic)) {
            orchestrator->stats.energy_blocked_events++;
            emit_event(orchestrator, PLASTICITY_EVENT_ENERGY_DEPLETED, 0, 0, atp, atp);
        }
    }

    /* 2. Calcium dynamics - update [Ca²⁺] */
    if (orchestrator->config.enabled.enable_calcium && orchestrator->calcium) {
        calcium_update(orchestrator->calcium, dt);
    }

    /* 3. STDP / Triplet STDP - handled via spike events */

    /* 3.5 Decay neuron firing rates during inactivity */
    for (size_t i = 0; i < orchestrator->num_neurons; i++) {
        neuron_entry_t* neuron = &orchestrator->neurons[i];
        if (!neuron->active) continue;

        /* Exponential decay with ~1 second time constant */
        float decay_rate = 0.001f;  /* 1/tau where tau = 1000ms */
        neuron->firing_rate *= expf(-decay_rate * dt);
    }

    /* 3.6 Decay synapse activity rates during inactivity */
    for (size_t i = 0; i < orchestrator->num_synapses; i++) {
        synapse_entry_t* syn = &orchestrator->synapses[i];
        if (!syn->active) continue;

        /* Exponential decay with ~5 second time constant for orchestrator tracking */
        float syn_decay_rate = 0.0002f;  /* 1/tau where tau = 5000ms */
        syn->recent_activity_hz *= expf(-syn_decay_rate * dt);
    }

    /* 4. Heterosynaptic - apply neighbor depression for recent potentiations */
    if (orchestrator->config.enabled.enable_heterosynaptic && orchestrator->heterosynaptic) {
        /* Depression is applied during LTP events */
    }

    /* 5. BCM - apply threshold-based selectivity */
    if (orchestrator->config.enabled.enable_bcm) {
        bcm_params_t bcm_params = bcm_params_cortical();

        for (size_t i = 0; i < orchestrator->num_synapses; i++) {
            synapse_entry_t* syn = &orchestrator->synapses[i];
            if (!syn->active) continue;

            /* Update BCM threshold */
            bcm_update_threshold(&syn->bcm_state, syn->recent_activity_hz, dt, &bcm_params);
        }

        /* Update neuron thresholds based on their firing rates (metaplasticity) */
        for (size_t i = 0; i < orchestrator->num_neurons; i++) {
            neuron_entry_t* neuron = &orchestrator->neurons[i];
            if (!neuron->active) continue;

            /* Threshold slides with average activity (BCM rule) */
            float rate = neuron->firing_rate;
            float tau = bcm_params.threshold_time_constant;

            /* Exponential sliding: dθ/dt = (rate² - θ) / tau */
            float dtheta = ((rate * rate) - neuron->bcm_threshold) * (dt / tau);
            neuron->bcm_threshold += dtheta;

            /* Clamp threshold */
            if (neuron->bcm_threshold < bcm_params.min_threshold)
                neuron->bcm_threshold = bcm_params.min_threshold;
            if (neuron->bcm_threshold > bcm_params.max_threshold)
                neuron->bcm_threshold = bcm_params.max_threshold;
        }
    }

    /* 6. Homeostatic - maintain target firing rates (periodic) */
    if (orchestrator->config.enabled.enable_homeostatic) {
        if (orchestrator->current_time_ms - orchestrator->last_homeostatic_time >=
            orchestrator->config.homeostatic_interval_ms) {

            /* Apply synaptic scaling to each neuron */
            synaptic_scaling_params_t scale_params = homeostatic_scaling_params_default();

            for (size_t i = 0; i < orchestrator->num_neurons; i++) {
                neuron_entry_t* neuron = &orchestrator->neurons[i];
                if (!neuron->active) continue;

                /* Update rate estimate */
                synaptic_scaling_update_rate(&neuron->scaling_state, false, dt, &scale_params);

                /* Compute scaling factor */
                float factor = synaptic_scaling_compute_factor(&neuron->scaling_state, &scale_params);

                if (fabsf(factor - 1.0f) > 0.01f) {
                    /* Apply scaling to input synapses */
                    emit_event(orchestrator, PLASTICITY_EVENT_HOMEOSTATIC_SCALE,
                              0, neuron->id, 1.0f, factor);
                }
            }

            orchestrator->last_homeostatic_time = orchestrator->current_time_ms;
        }
    }

    /* 7. Metaplasticity - update sliding thresholds */
    /* Note: metaplasticity controller was pre-allocated with MAX_SYNAPSES entries,
     * so we need to provide a full-size activities array to avoid buffer overread */
    if (orchestrator->config.enabled.enable_metaplasticity && orchestrator->metaplasticity) {
        /* Allocate full-size array (controller expects MAX_SYNAPSES entries) */
        float* activities = (float*)nimcp_calloc(MAX_SYNAPSES, sizeof(float));
        if (activities) {
            /* Fill in activity values for registered synapses only */
            for (size_t i = 0; i < orchestrator->num_synapses && i < MAX_SYNAPSES; i++) {
                activities[i] = orchestrator->synapses[i].recent_activity_hz;
            }

            metaplasticity_controller_update_all(
                orchestrator->metaplasticity,
                activities,
                &orchestrator->neuromod_levels,
                dt
            );

            nimcp_free(activities);
        }
    }

    /* 8. Protein synthesis - tag capture for consolidation (periodic) */
    if (orchestrator->config.enabled.enable_protein_synthesis && orchestrator->protein_synthesis) {
        protein_synthesis_update(orchestrator->protein_synthesis, delta_ms);

        if (orchestrator->current_time_ms - orchestrator->last_consolidation_time >=
            orchestrator->config.consolidation_interval_ms) {

            /* Attempt consolidation for tagged synapses */
            for (size_t i = 0; i < orchestrator->num_synapses; i++) {
                synapse_entry_t* syn = &orchestrator->synapses[i];
                if (!syn->active || !syn->consolidation_tagged) continue;

                if (protein_synthesis_can_consolidate(orchestrator->protein_synthesis, syn->id)) {
                    if (protein_synthesis_consolidate_synapse(orchestrator->protein_synthesis, syn->id) == 0) {
                        syn->consolidation_tagged = false;
                        emit_event(orchestrator, PLASTICITY_EVENT_CONSOLIDATION,
                                  syn->id, 0, syn->weight, syn->weight);
                    }
                }
            }

            orchestrator->last_consolidation_time = orchestrator->current_time_ms;
        }
    }

    /* 9. Structural plasticity - spine dynamics */
    if (orchestrator->config.enabled.enable_structural && orchestrator->structural) {
        structural_plasticity_update(orchestrator->structural, dt / 1000.0f);
    }

    /* 10. Astrocyte - glial modulation */
    if (orchestrator->config.enabled.enable_astrocyte && orchestrator->astrocyte) {
        /* Update astrocytes with average synaptic activity */
        float avg_activity = 0.0f;
        for (size_t i = 0; i < orchestrator->num_synapses; i++) {
            avg_activity += orchestrator->synapses[i].recent_activity_hz;
        }
        if (orchestrator->num_synapses > 0) {
            avg_activity /= (float)orchestrator->num_synapses;
        }

        for (uint32_t a = 0; a < MAX_ASTROCYTES && a < astrocyte_plasticity_get_num_astrocytes(orchestrator->astrocyte); a++) {
            astrocyte_plasticity_update(orchestrator->astrocyte, a, avg_activity / 100.0f, delta_ms);
        }
    }

    /* Update weight statistics */
    update_weight_statistics(orchestrator);

    /* Post-update callbacks */
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (orchestrator->post_update_callbacks[i].active) {
            orchestrator->post_update_callbacks[i].callback(
                orchestrator,
                delta_ms,
                orchestrator->post_update_callbacks[i].user_data
            );
        }
    }

    orchestrator->stats.total_updates++;
    orchestrator->stats.last_update_ms = orchestrator->current_time_ms;

    nimcp_platform_mutex_unlock(orchestrator->mutex);

    return 0;
}

/* ============================================================================
 * Spike Event Handling
 * ============================================================================ */

int plasticity_orchestrator_pre_spike(
    plasticity_orchestrator_t* orchestrator,
    uint32_t synapse_id,
    uint64_t timestamp_ms
) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in pre_spike");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    synapse_entry_t* syn = get_or_create_synapse(orchestrator, synapse_id);
    if (!syn) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return -1;
    }

    syn->last_pre_spike_time = timestamp_ms;
    syn->total_pre_spikes++;

    /* Update activity estimate */
    float time_since_last = (float)(timestamp_ms - syn->last_pre_spike_time);
    if (time_since_last > 0) {
        syn->recent_activity_hz = 1000.0f / time_since_last;  /* Convert to Hz */
    }

    /* Check metabolic constraints */
    bool can_ltp = true;
    bool can_ltd = true;
    if (orchestrator->config.enabled.enable_metabolic && orchestrator->metabolic) {
        can_ltp = metabolic_plasticity_can_ltp(orchestrator->metabolic);
        can_ltd = metabolic_plasticity_can_ltd(orchestrator->metabolic);
    }

    /* Compute modulation factor from sleep and immune states */
    float modulation = orchestrator->sleep_modulation_factor *
                       orchestrator->immune_modulation_factor;

    /* Process through triplet STDP if enabled */
    if (orchestrator->config.enabled.enable_triplet_stdp && syn->triplet_stdp && can_ltd) {
        float old_weight = syn->triplet_stdp->weight;
        triplet_stdp_pre_spike(syn->triplet_stdp, (float)timestamp_ms);

        float raw_dw = syn->triplet_stdp->weight - old_weight;
        float dw = raw_dw * modulation;  /* Scale by modulation factor */

        if (raw_dw < -0.001f) {
            /* LTD occurred - apply modulated weight change */
            syn->triplet_stdp->weight = old_weight + dw;
            syn->weight = syn->triplet_stdp->weight;
            syn->ltd_accumulator += fabsf(dw);
            orchestrator->stats.ltd_count++;
            emit_event(orchestrator, PLASTICITY_EVENT_LTD,
                      synapse_id, 0, old_weight, syn->triplet_stdp->weight);

            /* Consume ATP for LTD */
            if (orchestrator->metabolic) {
                metabolic_plasticity_consume_atp(orchestrator->metabolic, METABOLIC_EVENT_LTD, fabsf(dw));
            }
        }
    }

    /* Trigger calcium influx */
    if (orchestrator->config.enabled.enable_calcium && orchestrator->calcium) {
        calcium_trigger_nmda_influx(orchestrator->calcium, 0.5f, -30.0f);
    }

    /* Update structural plasticity activity */
    if (orchestrator->config.enabled.enable_structural && orchestrator->structural && syn->structural_synapse_id > 0) {
        structural_plasticity_update_activity(orchestrator->structural, syn->structural_synapse_id, timestamp_ms);
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int plasticity_orchestrator_post_spike(
    plasticity_orchestrator_t* orchestrator,
    uint32_t neuron_id,
    uint64_t timestamp_ms
) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in post_spike");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    neuron_entry_t* neuron = get_or_create_neuron(orchestrator, neuron_id);
    if (!neuron) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return -1;
    }

    /* Update neuron firing rate estimate */
    neuron->firing_rate = (neuron->firing_rate * NIMCP_EMA_WEIGHT_SLOW) + NIMCP_EMA_WEIGHT_FAST;

    /* Check metabolic constraints */
    bool can_ltp = true;
    if (orchestrator->config.enabled.enable_metabolic && orchestrator->metabolic) {
        can_ltp = metabolic_plasticity_can_ltp(orchestrator->metabolic);
    }

    /* Get calcium-dependent learning rate if enabled */
    float ca_lr = 1.0f;
    if (orchestrator->config.enabled.enable_calcium && orchestrator->calcium) {
        ca_lr = calcium_compute_learning_rate(orchestrator->calcium);
    }

    /* Compute modulation factor from sleep and immune states */
    float modulation = orchestrator->sleep_modulation_factor *
                       orchestrator->immune_modulation_factor;

    /* Process LTP for all active synapses */
    for (size_t i = 0; i < orchestrator->num_synapses; i++) {
        synapse_entry_t* syn = &orchestrator->synapses[i];
        if (!syn->active) continue;

        if (orchestrator->config.enabled.enable_triplet_stdp && syn->triplet_stdp && can_ltp) {
            float old_weight = syn->triplet_stdp->weight;
            triplet_stdp_post_spike(syn->triplet_stdp, (float)timestamp_ms);

            float raw_dw = syn->triplet_stdp->weight - old_weight;
            float dw = raw_dw * modulation;  /* Scale by modulation factor */

            if (raw_dw > 0.001f) {
                /* LTP occurred - apply modulated weight change */
                syn->triplet_stdp->weight = old_weight + dw;
                syn->weight = syn->triplet_stdp->weight;
                syn->ltp_accumulator += dw;
                orchestrator->stats.ltp_count++;
                emit_event(orchestrator, PLASTICITY_EVENT_LTP,
                          syn->id, neuron_id, old_weight, syn->triplet_stdp->weight);

                /* Consume ATP for LTP */
                if (orchestrator->metabolic) {
                    metabolic_plasticity_consume_atp(orchestrator->metabolic, METABOLIC_EVENT_LTP, dw);
                }

                /* Apply heterosynaptic depression to neighbors */
                if (orchestrator->config.enabled.enable_heterosynaptic && orchestrator->heterosynaptic) {
                    hetero_apply_depression(orchestrator->heterosynaptic, syn->id, dw, timestamp_ms);
                }

                /* Set synaptic tag if strong enough */
                if (dw > 0.1f && orchestrator->config.enabled.enable_protein_synthesis &&
                    orchestrator->protein_synthesis) {
                    protein_synthesis_set_tag(orchestrator->protein_synthesis, syn->id, dw);
                    syn->consolidation_tagged = true;
                }
            }
        }
    }

    /* Trigger calcium spike for postsynaptic event */
    if (orchestrator->config.enabled.enable_calcium && orchestrator->calcium) {
        calcium_set_concentration(orchestrator->calcium, 1.0f);  /* Strong Ca spike */
        emit_event(orchestrator, PLASTICITY_EVENT_CALCIUM_SPIKE, 0, neuron_id, 0.1f, 1.0f);
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int plasticity_orchestrator_reward(
    plasticity_orchestrator_t* orchestrator,
    float reward_magnitude,
    uint64_t timestamp_ms
) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in reward");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    /* Update dopamine level based on reward */
    orchestrator->neuromod_levels.dopamine = fmaxf(0.0f, fminf(1.0f,
        orchestrator->neuromod_levels.dopamine + reward_magnitude * 0.3f));

    /* Apply eligibility trace-based weight updates */
    if (orchestrator->config.enabled.enable_eligibility) {
        for (size_t i = 0; i < orchestrator->num_synapses; i++) {
            synapse_entry_t* syn = &orchestrator->synapses[i];
            if (!syn->active) continue;

            if (syn->eligibility_trace > 0.01f) {
                float dw = syn->eligibility_trace * reward_magnitude *
                          orchestrator->config.global_learning_rate;

                float old_weight = syn->weight;
                syn->weight = fmaxf(syn->w_min, fminf(syn->w_max, syn->weight + dw));

                if (syn->triplet_stdp) {
                    syn->triplet_stdp->weight = syn->weight;
                }

                if (dw > 0) {
                    emit_event(orchestrator, PLASTICITY_EVENT_LTP,
                              syn->id, 0, old_weight, syn->weight);
                } else if (dw < 0) {
                    emit_event(orchestrator, PLASTICITY_EVENT_LTD,
                              syn->id, 0, old_weight, syn->weight);
                }

                /* Decay eligibility trace */
                syn->eligibility_trace *= 0.95f;
            }
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

/* ============================================================================
 * Module Access Functions
 * ============================================================================ */

float plasticity_orchestrator_get_weight(
    const plasticity_orchestrator_t* orchestrator,
    uint32_t synapse_id
) {
    if (!orchestrator) return NAN;

    for (size_t i = 0; i < orchestrator->num_synapses; i++) {
        if (orchestrator->synapses[i].id == synapse_id &&
            orchestrator->synapses[i].active) {
            return orchestrator->synapses[i].weight;
        }
    }
    return NAN;
}

int plasticity_orchestrator_set_weight(
    plasticity_orchestrator_t* orchestrator,
    uint32_t synapse_id,
    float weight
) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in set_weight");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    synapse_entry_t* syn = get_or_create_synapse(orchestrator, synapse_id);
    if (!syn) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return -1;
    }

    syn->weight = fmaxf(syn->w_min, fminf(syn->w_max, weight));
    if (syn->triplet_stdp) {
        syn->triplet_stdp->weight = syn->weight;
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

float plasticity_orchestrator_get_atp_level(
    const plasticity_orchestrator_t* orchestrator
) {
    if (!orchestrator) return -1.0f;

    if (orchestrator->metabolic) {
        return metabolic_plasticity_get_atp_level(orchestrator->metabolic);
    }
    return 100.0f;  /* Default full ATP */
}

float plasticity_orchestrator_get_calcium(
    const plasticity_orchestrator_t* orchestrator,
    uint32_t compartment_id
) {
    if (!orchestrator) return -1.0f;
    (void)compartment_id;

    if (orchestrator->calcium) {
        return calcium_get_concentration(orchestrator->calcium);
    }
    return 0.1f;  /* Default resting calcium (μM) */
}

float plasticity_orchestrator_get_threshold(
    const plasticity_orchestrator_t* orchestrator,
    uint32_t neuron_id
) {
    if (!orchestrator) return -1.0f;

    for (size_t i = 0; i < orchestrator->num_neurons; i++) {
        if (orchestrator->neurons[i].id == neuron_id &&
            orchestrator->neurons[i].active) {
            return orchestrator->neurons[i].bcm_threshold;
        }
    }
    return 0.5f;  /* Default threshold */
}

/* ============================================================================
 * Statistics and Monitoring
 * ============================================================================ */

int plasticity_orchestrator_get_stats(
    const plasticity_orchestrator_t* orchestrator,
    plasticity_stats_t* stats
) {
    if (!orchestrator || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in get_stats");
        return -1;
    }

    *stats = orchestrator->stats;
    return 0;
}

int plasticity_orchestrator_reset_stats(plasticity_orchestrator_t* orchestrator) {
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Orchestrator is NULL in reset_stats");
        return -1;
    }

    nimcp_platform_mutex_lock(orchestrator->mutex);

    memset(&orchestrator->stats, 0, sizeof(plasticity_stats_t));
    orchestrator->stats.mean_weight = 0.5f;
    orchestrator->stats.mean_atp = 100.0f;
    orchestrator->stats.min_atp = 100.0f;

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

/* ============================================================================
 * State Persistence
 * ============================================================================ */

int plasticity_orchestrator_serialize(
    const plasticity_orchestrator_t* orchestrator,
    uint8_t* buffer,
    size_t buffer_size,
    size_t* bytes_written
) {
    if (!orchestrator || !buffer || !bytes_written) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL pointer in serialize");
        return -1;
    }

    /* Header: version, num_synapses, num_neurons */
    size_t required = sizeof(uint32_t) * 3 +
                     sizeof(synapse_entry_t) * orchestrator->num_synapses +
                     sizeof(neuron_entry_t) * orchestrator->num_neurons +
                     sizeof(plasticity_stats_t);

    if (buffer_size < required) {
        *bytes_written = 0;
        return -1;
    }

    uint8_t* ptr = buffer;

    /* Write version */
    *(uint32_t*)ptr = 1;
    ptr += sizeof(uint32_t);

    /* Write counts */
    *(uint32_t*)ptr = (uint32_t)orchestrator->num_synapses;
    ptr += sizeof(uint32_t);
    *(uint32_t*)ptr = (uint32_t)orchestrator->num_neurons;
    ptr += sizeof(uint32_t);

    /* Write synapse weights (simplified) */
    for (size_t i = 0; i < orchestrator->num_synapses; i++) {
        *(float*)ptr = orchestrator->synapses[i].weight;
        ptr += sizeof(float);
        *(uint32_t*)ptr = orchestrator->synapses[i].id;
        ptr += sizeof(uint32_t);
    }

    /* Write stats */
    memcpy(ptr, &orchestrator->stats, sizeof(plasticity_stats_t));
    ptr += sizeof(plasticity_stats_t);

    *bytes_written = ptr - buffer;
    return 0;
}

int plasticity_orchestrator_deserialize(
    plasticity_orchestrator_t* orchestrator,
    const uint8_t* buffer,
    size_t buffer_size
) {
    if (!orchestrator || !buffer || buffer_size < sizeof(uint32_t) * 3) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Invalid parameters in deserialize");
        return -1;
    }

    const uint8_t* ptr = buffer;

    /* Read version */
    uint32_t version = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);
    if (version != 1) return -1;

    /* Read counts */
    uint32_t num_synapses = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);
    uint32_t num_neurons = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);

    (void)num_neurons;  /* Not used in simplified deserialization */

    /* Read synapse weights */
    nimcp_platform_mutex_lock(orchestrator->mutex);

    for (uint32_t i = 0; i < num_synapses && i < orchestrator->synapse_capacity; i++) {
        float weight = *(float*)ptr;
        ptr += sizeof(float);
        uint32_t id = *(uint32_t*)ptr;
        ptr += sizeof(uint32_t);

        synapse_entry_t* syn = get_or_create_synapse(orchestrator, id);
        if (syn) {
            syn->weight = weight;
            if (syn->triplet_stdp) {
                syn->triplet_stdp->weight = weight;
            }
        }
    }

    /* Read stats */
    if (ptr + sizeof(plasticity_stats_t) <= buffer + buffer_size) {
        memcpy(&orchestrator->stats, ptr, sizeof(plasticity_stats_t));
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}
