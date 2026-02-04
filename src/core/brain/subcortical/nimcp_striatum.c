//=============================================================================
// nimcp_striatum.c - Striatum Implementation
//=============================================================================
/**
 * @file nimcp_striatum.c
 * @brief Striatum implementation with D1/D2 medium spiny neurons
 *
 * WHAT: Striatum model with distinct D1 (direct) and D2 (indirect) pathways
 * WHY:  Primary input nucleus of basal ganglia for action selection
 * HOW:  D1 MSNs facilitate actions (GO), D2 MSNs inhibit actions (NO-GO)
 *
 * @author NIMCP Development Team
 * @date 2025-12-22
 */

#include "core/brain/subcortical/nimcp_striatum.h"
#include "api/nimcp_api_exception.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(striatum)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_striatum_mesh_id = 0;
static mesh_participant_registry_t* g_striatum_mesh_registry = NULL;

nimcp_error_t striatum_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_striatum_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "striatum", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SUBCORTICAL);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "striatum";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_striatum_mesh_id);
    if (err == NIMCP_SUCCESS) g_striatum_mesh_registry = registry;
    return err;
}

void striatum_mesh_unregister(void) {
    if (g_striatum_mesh_registry && g_striatum_mesh_id != 0) {
        mesh_participant_unregister(g_striatum_mesh_registry, g_striatum_mesh_id);
        g_striatum_mesh_id = 0;
        g_striatum_mesh_registry = NULL;
    }
}


//=============================================================================
// Internal Helpers
//=============================================================================

/**
 * @brief Sigmoid activation function
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Initialize pathway neurons
 */
static int init_pathway(striatum_pathway_t* pathway, msn_type_t type,
                        uint32_t num_neurons, uint32_t num_actions,
                        float dopamine_gain, float baseline_firing) {
    pathway->type = type;
    pathway->num_neurons = num_neurons;
    pathway->num_actions = num_actions;
    pathway->dopamine_gain = dopamine_gain;
    pathway->baseline_firing = baseline_firing;

    pathway->neurons = nimcp_calloc(num_neurons, sizeof(msn_neuron_t));
    if (!pathway->neurons) {
        return -1;
    }

    pathway->activations = nimcp_calloc(num_actions, sizeof(float));
    if (!pathway->activations) {
        nimcp_free(pathway->neurons);
        pathway->neurons = NULL;
        return -1;
    }

    /* Initialize neurons */
    uint32_t neurons_per_action = num_neurons / num_actions;
    for (uint32_t i = 0; i < num_neurons; i++) {
        pathway->neurons[i].id = i;
        pathway->neurons[i].type = type;
        pathway->neurons[i].membrane_potential = -70.0f;  /* mV */
        pathway->neurons[i].firing_rate = baseline_firing;
        pathway->neurons[i].dopamine_sensitivity = 1.0f;
        pathway->neurons[i].cortical_weight = 1.0f;
        pathway->neurons[i].action_id = i / neurons_per_action;
        if (pathway->neurons[i].action_id >= num_actions) {
            pathway->neurons[i].action_id = num_actions - 1;
        }
    }

    return 0;
}

/**
 * @brief Destroy pathway
 */
static void destroy_pathway(striatum_pathway_t* pathway) {
    if (pathway->neurons) {
        nimcp_free(pathway->neurons);
        pathway->neurons = NULL;
    }
    if (pathway->activations) {
        nimcp_free(pathway->activations);
        pathway->activations = NULL;
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

void striatum_default_config(striatum_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "striatum_default_config: config is NULL");
        return;
    }

    config->neurons_per_pathway = STRIATUM_DEFAULT_NEURONS;
    config->num_actions = 8;
    config->d1_dopamine_gain = STRIATUM_D1_DOPAMINE_GAIN;
    config->d2_dopamine_gain = STRIATUM_D2_DOPAMINE_GAIN;
    config->baseline_firing = STRIATUM_BASELINE_FIRING;
    config->enable_lateral_inhibition = true;
    config->lateral_inhibition_strength = 0.1f;
}

striatum_t* striatum_create(const striatum_config_t* config) {
    striatum_config_t cfg;
    if (config) {
        cfg = *config;
    } else {
        striatum_default_config(&cfg);
    }

    striatum_t* striatum = nimcp_malloc(sizeof(striatum_t));
    if (!striatum) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "striatum_create: failed to allocate striatum");
        return NULL;
    }
    memset(striatum, 0, sizeof(striatum_t));

    striatum->config = cfg;
    striatum->dopamine_level = 0.5f;  /* Baseline */

    /* Initialize D1 (direct) pathway */
    if (init_pathway(&striatum->direct, MSN_TYPE_D1,
                     cfg.neurons_per_pathway, cfg.num_actions,
                     cfg.d1_dopamine_gain, cfg.baseline_firing) != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "striatum_create: failed to initialize D1 pathway");
        nimcp_free(striatum);
        return NULL;
    }

    /* Initialize D2 (indirect) pathway */
    if (init_pathway(&striatum->indirect, MSN_TYPE_D2,
                     cfg.neurons_per_pathway, cfg.num_actions,
                     cfg.d2_dopamine_gain, cfg.baseline_firing) != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "striatum_create: failed to initialize D2 pathway");
        destroy_pathway(&striatum->direct);
        nimcp_free(striatum);
        return NULL;
    }

    /* Initialize lateral inhibition matrix if enabled */
    if (cfg.enable_lateral_inhibition) {
        uint32_t n = cfg.neurons_per_pathway;
        striatum->inhibition_matrix = nimcp_calloc(n * n, sizeof(float));
        if (!striatum->inhibition_matrix) {
            NIMCP_LOGGING_WARN("Failed to allocate lateral inhibition matrix");
        } else {
            /* Initialize with lateral inhibition between neurons of same type */
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    if (i != j) {
                        striatum->inhibition_matrix[i * n + j] = cfg.lateral_inhibition_strength;
                    }
                }
            }
            striatum->lateral_inhibition_enabled = true;
        }
    }

    /* Initialize mutex */
    striatum->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!striatum->mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "striatum_create: failed to allocate mutex");
        destroy_pathway(&striatum->direct);
        destroy_pathway(&striatum->indirect);
        if (striatum->inhibition_matrix) nimcp_free(striatum->inhibition_matrix);
        nimcp_free(striatum);
        return NULL;
    }
    nimcp_mutex_init(striatum->mutex, NULL);

    NIMCP_LOGGING_DEBUG("Created striatum with %u neurons per pathway, %u actions",
                        cfg.neurons_per_pathway, cfg.num_actions);

    return striatum;
}

void striatum_destroy(striatum_t* striatum) {
    if (!striatum) return;

    // Don't lock mutex during destruction - object is being destroyed
    // Other threads should not be accessing a dying object
    destroy_pathway(&striatum->direct);
    destroy_pathway(&striatum->indirect);

    if (striatum->inhibition_matrix) {
        nimcp_free(striatum->inhibition_matrix);
    }

    nimcp_mutex_free(striatum->mutex);

    nimcp_free(striatum);

    NIMCP_LOGGING_DEBUG("Destroyed striatum");
}

int striatum_reset(striatum_t* striatum) {
    if (!striatum) return -1;

    nimcp_mutex_lock(striatum->mutex);

    /* Reset D1 pathway */
    for (uint32_t i = 0; i < striatum->direct.num_neurons; i++) {
        striatum->direct.neurons[i].membrane_potential = -70.0f;
        striatum->direct.neurons[i].firing_rate = striatum->config.baseline_firing;
    }
    memset(striatum->direct.activations, 0,
           striatum->direct.num_actions * sizeof(float));

    /* Reset D2 pathway */
    for (uint32_t i = 0; i < striatum->indirect.num_neurons; i++) {
        striatum->indirect.neurons[i].membrane_potential = -70.0f;
        striatum->indirect.neurons[i].firing_rate = striatum->config.baseline_firing;
    }
    memset(striatum->indirect.activations, 0,
           striatum->indirect.num_actions * sizeof(float));

    striatum->dopamine_level = 0.5f;

    nimcp_mutex_unlock(striatum->mutex);

    return 0;
}

//=============================================================================
// Processing Functions
//=============================================================================

int striatum_process_input(striatum_t* striatum, const float* cortical_input,
                           float dopamine) {
    if (!striatum || !cortical_input) return -1;

    nimcp_mutex_lock(striatum->mutex);

    striatum->dopamine_level = dopamine;
    float da_centered = dopamine - 0.5f;  /* Center around baseline */

    /* Process D1 (direct) pathway */
    /* D1 receptors: dopamine increases excitability */
    for (uint32_t a = 0; a < striatum->direct.num_actions; a++) {
        float input = cortical_input[a];

        /* D1 modulation: dopamine enhances GO signal */
        float da_mod = 1.0f + (da_centered * striatum->direct.dopamine_gain);

        /* Compute activation */
        float activation = sigmoid((input * da_mod - 0.3f) * 5.0f);
        striatum->direct.activations[a] = activation;
    }

    /* Process D2 (indirect) pathway */
    /* D2 receptors: dopamine decreases excitability */
    for (uint32_t a = 0; a < striatum->indirect.num_actions; a++) {
        float input = cortical_input[a];

        /* D2 modulation: dopamine reduces NO-GO signal */
        float da_mod = 1.0f - (da_centered * striatum->indirect.dopamine_gain);

        /* Compute activation */
        float activation = sigmoid((input * da_mod - 0.3f) * 5.0f);
        striatum->indirect.activations[a] = activation;
    }

    /* Update statistics */
    float d1_sum = 0, d2_sum = 0;
    for (uint32_t a = 0; a < striatum->direct.num_actions; a++) {
        d1_sum += striatum->direct.activations[a];
        d2_sum += striatum->indirect.activations[a];
    }
    striatum->stats.avg_d1_firing = d1_sum / striatum->direct.num_actions;
    striatum->stats.avg_d2_firing = d2_sum / striatum->indirect.num_actions;
    if (d2_sum > 0.001f) {
        striatum->stats.d1_d2_ratio = d1_sum / d2_sum;
    } else {
        striatum->stats.d1_d2_ratio = d1_sum > 0 ? 10.0f : 1.0f;
    }

    nimcp_mutex_unlock(striatum->mutex);

    return 0;
}

int striatum_get_d1_output(const striatum_t* striatum, float* output) {
    if (!striatum || !output) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)striatum->mutex);
    memcpy(output, striatum->direct.activations,
           striatum->direct.num_actions * sizeof(float));
    nimcp_mutex_unlock((nimcp_mutex_t*)striatum->mutex);

    return 0;
}

int striatum_get_d2_output(const striatum_t* striatum, float* output) {
    if (!striatum || !output) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)striatum->mutex);
    memcpy(output, striatum->indirect.activations,
           striatum->indirect.num_actions * sizeof(float));
    nimcp_mutex_unlock((nimcp_mutex_t*)striatum->mutex);

    return 0;
}

float striatum_get_d1_activation(const striatum_t* striatum, uint32_t action_id) {
    if (!striatum || action_id >= striatum->direct.num_actions) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)striatum->mutex);
    float activation = striatum->direct.activations[action_id];
    nimcp_mutex_unlock((nimcp_mutex_t*)striatum->mutex);

    return activation;
}

float striatum_get_d2_activation(const striatum_t* striatum, uint32_t action_id) {
    if (!striatum || action_id >= striatum->indirect.num_actions) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)striatum->mutex);
    float activation = striatum->indirect.activations[action_id];
    nimcp_mutex_unlock((nimcp_mutex_t*)striatum->mutex);

    return activation;
}

int striatum_set_dopamine(striatum_t* striatum, float level) {
    if (!striatum) return -1;

    nimcp_mutex_lock(striatum->mutex);
    striatum->dopamine_level = fmaxf(0.0f, fminf(1.0f, level));
    nimcp_mutex_unlock(striatum->mutex);

    return 0;
}

int striatum_step(striatum_t* striatum, float dt) {
    if (!striatum) return -1;

    /* Currently no temporal dynamics beyond immediate processing */
    (void)dt;

    return 0;
}

int striatum_update_weights(striatum_t* striatum, uint32_t action_id,
                            float delta_d1, float delta_d2) {
    if (!striatum || action_id >= striatum->config.num_actions) return -1;

    nimcp_mutex_lock(striatum->mutex);

    /* Update D1 neuron weights for this action */
    uint32_t neurons_per_action = striatum->direct.num_neurons / striatum->direct.num_actions;
    uint32_t start_idx = action_id * neurons_per_action;
    uint32_t end_idx = start_idx + neurons_per_action;

    for (uint32_t i = start_idx; i < end_idx && i < striatum->direct.num_neurons; i++) {
        striatum->direct.neurons[i].cortical_weight += delta_d1;
        striatum->direct.neurons[i].cortical_weight =
            fmaxf(0.0f, fminf(2.0f, striatum->direct.neurons[i].cortical_weight));
    }

    /* Update D2 neuron weights for this action */
    neurons_per_action = striatum->indirect.num_neurons / striatum->indirect.num_actions;
    start_idx = action_id * neurons_per_action;
    end_idx = start_idx + neurons_per_action;

    for (uint32_t i = start_idx; i < end_idx && i < striatum->indirect.num_neurons; i++) {
        striatum->indirect.neurons[i].cortical_weight += delta_d2;
        striatum->indirect.neurons[i].cortical_weight =
            fmaxf(0.0f, fminf(2.0f, striatum->indirect.neurons[i].cortical_weight));
    }

    nimcp_mutex_unlock(striatum->mutex);

    return 0;
}

int striatum_get_stats(const striatum_t* striatum, striatum_stats_t* stats) {
    if (!striatum || !stats) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)striatum->mutex);
    *stats = striatum->stats;
    nimcp_mutex_unlock((nimcp_mutex_t*)striatum->mutex);

    return 0;
}
