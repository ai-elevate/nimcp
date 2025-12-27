//=============================================================================
// nimcp_neuralnet_homeostasis.c - Homeostatic Plasticity Mechanisms
//=============================================================================

#include "core/neuralnet/nimcp_neuralnet_homeostasis.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE "neuralnet_homeostasis"

// Constants
#define CALCIUM_DECAY_RATE 0.1f
#define META_PLASTICITY_RATE NIMCP_DEFAULT_DECAY_RATE
#define HOMEOSTATIC_DECAY 0.999f
#define MAX_SYNAPTIC_STRENGTH 10.0f
#define NORMALIZATION_INTERVAL 1000

// External structure definition
struct neural_network_struct {
    neuron_t* neurons;
    uint32_t num_neurons;
    uint32_t capacity;
    uint64_t current_time;
    uint64_t network_time;
    float global_activity;
    float network_stability;
    float learning_momentum;
    float last_avg_weight;
    uint64_t last_maintenance;
    void* neuromodulator_system;
    float* global_state;
    uint32_t global_state_size;
    void* glial_integration;
    void* axon_network;
    void* bio_ctx;
    bool bio_async_enabled;
    void* immune_system;  // brain_immune_system_t*
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

static void update_calcium_dynamics(neuron_t* neuron, uint64_t timestamp)
{
    if (!neuron)
        return;

    float dt = (timestamp > neuron->last_update) ?
               (float)(timestamp - neuron->last_update) : 1.0f;

    float decay = expf(-CALCIUM_DECAY_RATE * dt);
    neuron->calcium_concentration *= decay;

    if (neuron->last_spike > 0 && timestamp > neuron->last_spike) {
        float spike_dt = (float)(timestamp - neuron->last_spike);
        if (spike_dt < 100.0f) {
            neuron->calcium_concentration += expf(-spike_dt / 20.0f);
        }
    }

    if (neuron->calcium_concentration > 10.0f) {
        neuron->calcium_concentration = 10.0f;
    }
}

static void update_meta_plasticity(neuron_t* neuron, uint64_t timestamp)
{
    if (!neuron)
        return;

    float target = neuron->homeostatic.target_activity;
    float current = neuron->avg_activity;
    float error = target - current;

    neuron->homeostatic_factor += META_PLASTICITY_RATE * error;

    if (neuron->homeostatic_factor < 0.1f)
        neuron->homeostatic_factor = 0.1f;
    if (neuron->homeostatic_factor > 10.0f)
        neuron->homeostatic_factor = 10.0f;
}

static float compute_homeostatic_factor(neuron_t* neuron, float current_activity)
{
    if (!neuron)
        return 1.0f;

    float target = neuron->homeostatic.target_activity;
    float strength = neuron->homeostatic.strength;

    if (target < 1e-6f)
        return 1.0f;

    float ratio = current_activity / target;
    return 1.0f + strength * (1.0f - ratio);
}

static void normalize_synaptic_weights(neuron_t* neuron)
{
    if (!neuron || neuron->num_synapses == 0)
        return;

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        float w = neuron->synapses[i].weight;
        sum_sq += w * w;
    }

    if (sum_sq < 1e-10f)
        return;

    float norm = sqrtf(sum_sq);
    neuron->weight_norm = norm;

    float target_norm = neuron->oja_params.target_norm;
    if (norm > target_norm * 1.5f) {
        float scale = target_norm / norm;
        for (uint32_t i = 0; i < neuron->num_synapses; i++) {
            neuron->synapses[i].weight *= scale;
        }
    }
}

//=============================================================================
// Immune System Integration Implementation
// NOTE: Base homeostasis functions (apply_homeostasis, maintain_homeostasis,
//       adapt_threshold_by_activity, maintain) are defined in nimcp_neuralnet.c
//=============================================================================

bool neural_network_apply_immune_inflammation(neural_network_t network,
                                             float inflammation_level,
                                             uint32_t region_id)
{
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in apply_immune_inflammation");
        return false;
    }

    if (inflammation_level < 0.0f || inflammation_level > 1.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid inflammation_level %.3f (must be 0.0-1.0)", inflammation_level);
        return false;
    }

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        if (region_id != 0 && neuron->id != region_id) {
            continue;
        }

        if (neuron->homeostatic.baseline_target_activity < 1e-6f) {
            neuron->homeostatic.baseline_target_activity = neuron->homeostatic.target_activity;
        }

        neuron->homeostatic.inflammation_modulation = inflammation_level;
        float fever_boost = 1.0f + (inflammation_level * 0.5f);
        neuron->homeostatic.target_activity =
            neuron->homeostatic.baseline_target_activity * fever_boost;

        if (inflammation_level > 0.1f && neuron->homeostatic.inflammation_start == 0) {
            neuron->homeostatic.inflammation_start = network->network_time;
        }
    }

    LOG_DEBUG(LOG_MODULE, "Applied inflammation %.3f to region %u",
              inflammation_level, region_id);
    return true;
}

bool neural_network_apply_anti_inflammatory(neural_network_t network,
                                           float il10_concentration,
                                           uint32_t region_id)
{
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in apply_anti_inflammatory");
        return false;
    }

    if (il10_concentration < 0.0f || il10_concentration > 1.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid il10_concentration %.3f", il10_concentration);
        return false;
    }

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        if (region_id != 0 && neuron->id != region_id) {
            continue;
        }

        float reduction = il10_concentration * neuron->homeostatic.inflammation_modulation;
        neuron->homeostatic.inflammation_modulation -= reduction;

        if (neuron->homeostatic.inflammation_modulation < 0.01f) {
            neuron->homeostatic.inflammation_modulation = 0.0f;
            neuron->homeostatic.inflammation_start = 0;
        }

        if (neuron->homeostatic.baseline_target_activity > 1e-6f) {
            float fever_boost = 1.0f + (neuron->homeostatic.inflammation_modulation * 0.5f);
            neuron->homeostatic.target_activity =
                neuron->homeostatic.baseline_target_activity * fever_boost;
        }

        neuron->homeostatic.metabolic_load *= (1.0f - il10_concentration * 0.5f);
    }

    LOG_DEBUG(LOG_MODULE, "Applied IL-10 %.3f to region %u", il10_concentration, region_id);
    return true;
}

bool neural_network_modulate_scaling_rate(neural_network_t network,
                                          uint32_t neuron_id,
                                          float cytokine_modulation)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    if (cytokine_modulation < -1.0f || cytokine_modulation > 1.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid cytokine_modulation %.3f", cytokine_modulation);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    neuron->homeostatic.cytokine_scaling_factor = cytokine_modulation;

    float scaling_multiplier = 1.0f + cytokine_modulation;
    if (scaling_multiplier < 0.1f) scaling_multiplier = 0.1f;
    if (scaling_multiplier > 2.0f) scaling_multiplier = 2.0f;

    neuron->homeostatic.strength *= scaling_multiplier;

    LOG_DEBUG(LOG_MODULE, "Modulated scaling rate for neuron %u by %.3f",
              neuron_id, cytokine_modulation);
    return true;
}

bool neural_network_apply_immune_metabolic_load(neural_network_t network,
                                                float metabolic_load,
                                                uint32_t region_id)
{
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in apply_immune_metabolic_load");
        return false;
    }

    if (metabolic_load < 0.0f || metabolic_load > 1.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid metabolic_load %.3f", metabolic_load);
        return false;
    }

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        if (region_id != 0 && neuron->id != region_id) {
            continue;
        }

        neuron->homeostatic.metabolic_load = metabolic_load;

        float plasticity_reduction = 1.0f - (metabolic_load * 0.5f);
        neuron->plasticity_rate *= plasticity_reduction;
        if (neuron->plasticity_rate < NIMCP_DEFAULT_DECAY_RATE) {
            neuron->plasticity_rate = NIMCP_DEFAULT_DECAY_RATE;
        }
    }

    LOG_DEBUG(LOG_MODULE, "Applied metabolic load %.3f to region %u",
              metabolic_load, region_id);
    return true;
}

bool neural_network_accumulate_allostatic_load(neural_network_t network,
                                               uint32_t neuron_id,
                                               uint64_t inflammation_duration,
                                               float inflammation_level)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    if (inflammation_level < 0.0f || inflammation_level > 1.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid inflammation_level %.3f", inflammation_level);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    float duration_factor = (float)inflammation_duration / 1000.0f;
    float load_increment = inflammation_level * duration_factor * 0.001f;
    neuron->homeostatic.allostatic_load += load_increment;

    float health_penalty = 1.0f - fminf(neuron->homeostatic.allostatic_load * 0.1f, 0.8f);
    neuron->homeostatic.strength *= health_penalty;

    LOG_DEBUG(LOG_MODULE, "Accumulated allostatic load %.6f for neuron %u (duration=%lu ms, level=%.3f)",
              neuron->homeostatic.allostatic_load, neuron_id,
              (unsigned long)inflammation_duration, inflammation_level);
    return true;
}

float neural_network_compute_homeostatic_health(neural_network_t network,
                                                uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return 0.0f;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    float activity_balance = 1.0f;
    if (neuron->homeostatic.target_activity > 1e-6f) {
        float activity_ratio = neuron->avg_activity / neuron->homeostatic.target_activity;
        activity_balance = 1.0f - fminf(fabsf(1.0f - activity_ratio), 1.0f);
    }

    float inflammation_penalty = 1.0f - (neuron->homeostatic.inflammation_modulation * 0.5f);
    float metabolic_penalty = 1.0f - (neuron->homeostatic.metabolic_load * 0.3f);
    float allostatic_penalty = expf(-neuron->homeostatic.allostatic_load);

    float health = activity_balance * inflammation_penalty * metabolic_penalty * allostatic_penalty;

    if (health < 0.0f) health = 0.0f;
    if (health > 1.0f) health = 1.0f;

    return health;
}

bool neural_network_connect_immune_system(neural_network_t network,
                                          brain_immune_system_t* immune_system)
{
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in connect_immune_system");
        return false;
    }

    network->immune_system = (void*)immune_system;

    LOG_INFO(LOG_MODULE, "Connected immune system to neural network");
    return true;
}
