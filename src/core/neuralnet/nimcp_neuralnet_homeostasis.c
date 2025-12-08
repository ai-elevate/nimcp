//=============================================================================
// nimcp_neuralnet_homeostasis.c - Homeostatic Plasticity Mechanisms
//=============================================================================

#include "core/neuralnet/nimcp_neuralnet_homeostasis.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE "neuralnet_homeostasis"

// Constants
#define CALCIUM_DECAY_RATE 0.1f
#define META_PLASTICITY_RATE 0.001f
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
// Public API Implementation
//=============================================================================

bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id,
                                     uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    update_calcium_dynamics(neuron, timestamp);
    update_meta_plasticity(neuron, timestamp);
    normalize_synaptic_weights(neuron);

    float current_activity = neuron->avg_activity;
    float factor = compute_homeostatic_factor(neuron, current_activity);

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        neuron->synapses[i].weight *= (1.0f + 0.001f * (factor - 1.0f));
    }

    LOG_DEBUG(LOG_MODULE, "Applied homeostasis to neuron %u, factor=%.3f", neuron_id, factor);
    return true;
}

void neural_network_maintain_homeostasis(neural_network_t network, uint64_t timestamp)
{
    if (!network) {
        LOG_ERROR(LOG_MODULE, "NULL network in maintain_homeostasis");
        return;
    }

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neural_network_apply_homeostasis(network, i, timestamp);
    }

    LOG_INFO(LOG_MODULE, "Maintained homeostasis for %u neurons", network->num_neurons);
}

bool neural_network_adapt_threshold_by_activity(neural_network_t network, uint32_t neuron_id,
                                                float activity_level)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    float target = neuron->homeostatic.target_activity;

    if (activity_level > target * 1.5f) {
        neuron->threshold *= 1.01f;
    } else if (activity_level < target * 0.5f) {
        neuron->threshold *= 0.99f;
    }

    if (neuron->threshold < 0.1f)
        neuron->threshold = 0.1f;
    if (neuron->threshold > 2.0f)
        neuron->threshold = 2.0f;

    return true;
}

void neural_network_maintain(neural_network_t network, uint64_t timestamp)
{
    if (!network)
        return;

    if (timestamp - network->last_maintenance < NORMALIZATION_INTERVAL) {
        return;
    }

    neural_network_maintain_homeostasis(network, timestamp);
    network->last_maintenance = timestamp;

    LOG_INFO(LOG_MODULE, "Network maintenance completed at t=%lu", (unsigned long)timestamp);
}
