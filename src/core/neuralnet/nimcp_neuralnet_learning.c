#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_neuralnet_learning.c - Learning Rules and Synaptic Plasticity
//=============================================================================
// NOTE: NEURALNET_LEARNING_SEPARATE is defined via CMake to guard duplicates
//       in nimcp_neuralnet.c when this file is compiled.

#include "core/neuralnet/nimcp_neuralnet_learning.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "plasticity/bcm/nimcp_bcm.h"
#include "plasticity/eligibility/nimcp_eligibility_trace.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "security/nimcp_security.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE "neuralnet_learning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(neuralnet_learning)

// Constants
#define TRACE_DECAY_RATE 0.05f
#define WEIGHT_UPDATE_THRESHOLD 1e-6f
#define NORM_THRESHOLD 1e-7f
#define EPSILON 1e-10f
#define MAX_SYNAPTIC_STRENGTH 10.0f
#define NORMALIZATION_INTERVAL 1000
#define ACTIVITY_THRESHOLD 0.01f

// External structure definition (opaque pointer access)
struct neural_network_struct {
    neuron_t* neurons;
    uint32_t num_neurons;
    uint32_t capacity;
    uint64_t current_time;
    uint64_t network_time;
    network_config_t config;  // Network configuration
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
    // Bio-async fields
    void* bio_ctx;
    bool bio_async_enabled;
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

static float compute_stdp_update(float dt, const stdp_params_t* params)
{
    if (!params || fabsf(dt) < 1e-6f)
        return 0.0f;

    float time_window = params->time_window;
    if (time_window < 1.0f)
        time_window = 20.0f;

    if (dt > 0.0f) {
        // Pre before post - potentiation
        return params->learning_rate * params->positive_factor * expf(-dt / time_window);
    } else {
        // Post before pre - depression
        return -params->learning_rate * params->negative_factor * expf(dt / time_window);
    }
}

static float compute_oja_weight_update(float pre_activity, float post_activity,
                                       float current_weight, const oja_params_t* params)
{
    if (!params)
        return 0.0f;

    float alpha = params->alpha;
    float forgetting = params->forgetting;

    // Oja's rule: Δw = α * y * (x - y * w)
    float hebbian_term = alpha * post_activity * pre_activity;
    float decay_term = alpha * post_activity * post_activity * current_weight;
    float forgetting_term = forgetting * current_weight;

    return hebbian_term - decay_term - forgetting_term;
}

static void update_synaptic_traces(neuron_t* neuron, uint64_t timestamp)
{
    if (!neuron)
        return;

    float dt = (timestamp > neuron->last_update) ?
               (float)(timestamp - neuron->last_update) : 1.0f;

    float decay = expf(-TRACE_DECAY_RATE * dt);

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        neuron->synapses[i].trace *= decay;
    }
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

    if (sum_sq < NORM_THRESHOLD)
        return;

    float norm = sqrtf(sum_sq);
    neuron->weight_norm = norm;

    float target_norm = neuron->oja_params.target_norm;
    if (norm > target_norm * 1.5f) {
        float scale = target_norm / norm;
        for (uint32_t i = 0; i < neuron->num_synapses; i++) {
            neuron->synapses[i].weight *= scale;
        }
        LOG_DEBUG(LOG_MODULE, "Normalized weights for neuron %u: %.3f -> %.3f",
                  neuron->id, norm, target_norm);
    }
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool neural_network_normalize_weights(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "neural_network_normalize_weights: network is NULL");
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    normalize_synaptic_weights(neuron);

    return true;
}

void neural_network_update_traces(neural_network_t network, uint32_t neuron_id,
                                 uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    update_synaptic_traces(neuron, timestamp);
}

float neural_network_get_weight_norm(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return 0.0f;
    }

    return network->neurons[neuron_id].weight_norm;
}

void neural_network_get_weight_statistics_ext(neural_network_t network, uint32_t neuron_id,
                                              float* mean, float* std_dev,
                                              float* min_weight, float* max_weight)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        if (mean) *mean = 0.0f;
        if (std_dev) *std_dev = 0.0f;
        if (min_weight) *min_weight = 0.0f;
        if (max_weight) *max_weight = 0.0f;
        return;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    if (neuron->num_synapses == 0) {
        if (mean) *mean = 0.0f;
        if (std_dev) *std_dev = 0.0f;
        if (min_weight) *min_weight = 0.0f;
        if (max_weight) *max_weight = 0.0f;
        return;
    }

    float sum = 0.0f;
    float min_w = neuron->synapses[0].weight;
    float max_w = neuron->synapses[0].weight;

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        float w = neuron->synapses[i].weight;
        sum += w;
        if (w < min_w) min_w = w;
        if (w > max_w) max_w = w;
    }

    float avg = sum / (float)neuron->num_synapses;

    float variance = 0.0f;
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        float diff = neuron->synapses[i].weight - avg;
        variance += diff * diff;
    }
    variance /= (float)neuron->num_synapses;

    if (mean) *mean = avg;
    if (std_dev) *std_dev = sqrtf(variance);
    if (min_weight) *min_weight = min_w;
    if (max_weight) *max_weight = max_w;
}

//=============================================================================
// Learning Rule Implementations
//=============================================================================

uint32_t neural_network_apply_oja(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    // Guard: Validate inputs
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return 0;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    uint32_t modified = 0;

    // Use avg_activity instead of current state (which may be 0 after spike reset)
    float x = neuron->avg_activity;  // Pre-synaptic average activity

    // Skip update if neuron is not active enough
    if (fabsf(x) < ACTIVITY_THRESHOLD) {
        return 0;
    }

    // Calculate weight updates using Oja's rule
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Get post-synaptic average activity (target of synapse)
        float y = network->neurons[syn->target_id].avg_activity;

        // Compute weight update using Oja's rule: Δw = α(y*x - y²*w)
        float delta_w = compute_oja_weight_update(x, y, syn->weight, &neuron->oja_params);

        // Apply weight update with meta-plasticity
        float new_weight = syn->weight + delta_w * syn->meta_plasticity;

        // Apply weight constraints
        new_weight = fmaxf(network->config.min_weight,
                          fminf(network->config.max_weight, new_weight));

        // Update weight if change is significant
        if (fabsf(new_weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
            syn->weight = new_weight;
            modified++;
        }

        // Update synaptic strength
        syn->strength = fminf(syn->strength * (1.0f + delta_w), MAX_SYNAPTIC_STRENGTH);
    }

    // Normalize weights periodically
    if (modified > 0 && (timestamp - neuron->last_update) > NORMALIZATION_INTERVAL) {
        normalize_synaptic_weights(neuron);
    }

    LOG_DEBUG(LOG_MODULE, "Applied Oja's rule to neuron %u, modified %u synapses",
              neuron_id, modified);
    return modified;
}

uint32_t neural_network_apply_stdp(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return 0;
    }

    neuron_t* pre_neuron = &network->neurons[neuron_id];
    uint32_t modified = 0;

    // Iterate over all outgoing synapses from this neuron
    for (uint32_t i = 0; i < pre_neuron->num_synapses; i++) {
        synapse_t* syn = &pre_neuron->synapses[i];

        // Get postsynaptic neuron (target of this synapse)
        const neuron_t* post_neuron = &network->neurons[syn->target_id];

        // Guard: Skip if either neuron never spiked
        if (pre_neuron->last_spike == 0 || post_neuron->last_spike == 0) {
            continue;
        }

        // Compute spike time difference (Δt = t_post - t_pre)
        int64_t dt = (int64_t)(post_neuron->last_spike) - (int64_t)(pre_neuron->last_spike);

        // Guard: Skip if time difference exceeds STDP window
        if (fabsf((float)dt) > pre_neuron->stdp_params.time_window) {
            continue;
        }

        // Compute STDP weight change factor
        float stdp_factor = compute_stdp_update((float)dt, &pre_neuron->stdp_params);

        // Scale STDP by learning rate, trace, and meta-plasticity
        float delta_w = pre_neuron->stdp_params.learning_rate * stdp_factor * syn->trace;

        // Apply weight update with meta-plasticity modulation
        float new_weight = syn->weight + delta_w * syn->meta_plasticity;

        // Clamp weight to configured bounds
        new_weight = fmaxf(network->config.min_weight,
                          fminf(network->config.max_weight, new_weight));

        // Update weight if change is significant
        if (fabsf(new_weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
            syn->weight = new_weight;
            modified++;
        }

        // Apply BCM homeostatic plasticity after STDP (if enabled)
        if (syn->enable_bcm && syn->bcm) {
            bcm_params_t bcm_params = bcm_params_cortical();
            float dt_sec = (float)(timestamp - syn->last_active) / 1000000.0f;
            float pre_activity = syn->trace;
            float post_activity = post_neuron->state;

            bcm_apply_rule(syn->bcm, pre_activity, post_activity, dt_sec, &bcm_params);

            if (fabsf(syn->bcm->weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
                syn->weight = syn->bcm->weight;
            }
        }
    }

    LOG_DEBUG(LOG_MODULE, "Applied STDP to neuron %u, modified %u synapses",
              neuron_id, modified);
    return modified;
}

uint32_t neural_network_apply_lateral_inhibition(
    neural_network_t network,
    uint32_t output_start,
    uint32_t output_count,
    float inhibition_strength)
{
    // Guard clauses
    if (!network || output_count == 0) return 0;
    if (inhibition_strength <= 0.0f) return 0;
    if (inhibition_strength > 1.0f) inhibition_strength = 1.0f;

    // Find the winner (max activation neuron)
    float max_activation = -1e30f;
    uint32_t winner_idx = 0;

    for (uint32_t i = 0; i < output_count; i++) {
        uint32_t nid = output_start + i;
        if (nid >= network->num_neurons) break;
        float act = network->neurons[nid].state;
        if (act > max_activation) {
            max_activation = act;
            winner_idx = i;
        }
    }

    // Compute mean activation for reference
    float mean_activation = 0.0f;
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < output_count; i++) {
        uint32_t nid = output_start + i;
        if (nid >= network->num_neurons) break;
        mean_activation += network->neurons[nid].state;
        valid_count++;
    }
    if (valid_count == 0) return 0;
    mean_activation /= (float)valid_count;

    // Apply inhibition: suppress neurons below winner
    // Use soft WTA: scale non-winners toward mean, don't zero them out
    uint32_t modified = 0;
    for (uint32_t i = 0; i < output_count; i++) {
        if (i == winner_idx) continue;
        uint32_t nid = output_start + i;
        if (nid >= network->num_neurons) break;

        float old_state = network->neurons[nid].state;
        // Move non-winners toward mean by inhibition_strength fraction
        float new_state = old_state + inhibition_strength * (mean_activation - old_state);
        // Also reduce bias slightly to make inhibition persistent
        network->neurons[nid].bias -= inhibition_strength * 0.001f *
                                       fmaxf(0.0f, old_state - mean_activation);
        network->neurons[nid].state = new_state;

        if (fabsf(new_state - old_state) > 1e-6f) {
            modified++;
        }
    }

    // Boost winner slightly
    uint32_t winner_nid = output_start + winner_idx;
    if (winner_nid < network->num_neurons) {
        network->neurons[winner_nid].state *= (1.0f + inhibition_strength * 0.1f);
        // Clamp
        if (network->neurons[winner_nid].state > 1.0f)
            network->neurons[winner_nid].state = 1.0f;
    }

    LOG_DEBUG(LOG_MODULE, "Lateral inhibition: modified %u neurons, winner=%u, strength=%.3f",
              modified, winner_idx, inhibition_strength);
    return modified;
}

uint32_t neural_network_apply_reward_learning(neural_network_t network, float reward,
                                              float learning_rate, uint64_t current_time)
{
    // Guard: Validate parameters
    if (!network || reward < 0.0f || reward > 1.0f || learning_rate <= 0.0f) {
        LOG_ERROR(LOG_MODULE, "Invalid parameters: reward=%.3f, lr=%.3f", reward, learning_rate);
        return 0;
    }

    uint32_t total_modified = 0;

    // PRE-PASS: Set outgoing synapse traces from current neuron states
    // WHY: The forward pass (neural_network_forward) sets neuron->state via
    //      incoming_synapses but never updates outgoing synapse traces.
    //      Eligibility learning needs syn->trace > 0.1 to trigger updates.
    //      Set each neuron's outgoing synapse traces to |neuron->state| so
    //      eligibility-based credit assignment knows which synapses were active.
    for (uint32_t n = 0; n < network->num_neurons; n++) {
        float activity = fabsf(network->neurons[n].state);
        neuron_t* neuron = &network->neurons[n];
        for (uint32_t s = 0; s < neuron->num_synapses; s++) {
            neuron->synapses[s].trace = activity;
        }
    }

    // Iterate over all neurons in the network
    for (uint32_t neuron_id = 0; neuron_id < network->config.num_neurons; neuron_id++) {
        neuron_t* neuron = &network->neurons[neuron_id];

        // Apply STDP if enabled for this neuron
        if (neuron->learning_rule & LEARNING_STDP) {
            total_modified += neural_network_apply_stdp(network, neuron_id, current_time);
        }

        // Apply Oja's rule if enabled
        if (neuron->learning_rule & LEARNING_OJA) {
            total_modified += neural_network_apply_oja(network, neuron_id, current_time);
        }

        // Apply eligibility-trace-based learning with reward signal
        for (uint32_t syn_idx = 0; syn_idx < neuron->num_synapses; syn_idx++) {
            synapse_t* syn = &neuron->synapses[syn_idx];

            // Update eligibility traces
            if (syn->enable_eligibility && syn->eligibility) {
                eligibility_config_t elig_config = eligibility_default_config();
                elig_config.learning_rate = learning_rate;

                // Update trace based on synaptic activity
                if (syn->trace > 0.1f) {
                    eligibility_trace_update(syn->eligibility, &elig_config, current_time, syn->trace);
                } else {
                    eligibility_trace_decay(syn->eligibility, &elig_config, current_time);
                }

                // Get dopamine level from neuromodulator system (if available)
                float dopamine = 0.5f;
                if (network->neuromodulator_system) {
                    dopamine = neuromodulator_get_level((neuromodulator_system_t)network->neuromodulator_system,
                                                       NEUROMOD_DOPAMINE);
                }

                // Apply eligibility-based weight update
                float old_weight = syn->weight;
                eligibility_apply_reward(syn, syn->eligibility, &elig_config, reward, dopamine);

                // Biological security: Validate weight change
                if (!nimcp_security_validate_weight_change(old_weight, syn->weight,
                                                          NIMCP_MAX_WEIGHT_DELTA_PER_STEP)) {
                    syn->weight = old_weight;
                    continue;
                }

                // Clamp weights to valid range
                syn->weight = fmaxf(network->config.min_weight,
                                  fminf(network->config.max_weight, syn->weight));

                if (fabsf(syn->weight - old_weight) > WEIGHT_UPDATE_THRESHOLD) {
                    total_modified++;
                }
            }

            // Apply BCM homeostatic plasticity
            if (syn->enable_bcm && syn->bcm) {
                bcm_params_t bcm_params = bcm_params_cortical();
                const neuron_t* post_neuron = &network->neurons[syn->target_id];

                bcm_apply_rule(syn->bcm, neuron->state, post_neuron->state, 1.0f, &bcm_params);

                if (fabsf(syn->bcm->weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
                    syn->weight = syn->bcm->weight;
                    total_modified++;
                }
            }
        }
    }

    // POST-PASS: Sync outgoing synapse weights to incoming synapses
    // WHY: Learning modifies outgoing synapse weights (neuron->synapses[]),
    //      but neural_network_forward() reads from incoming_synapses[].
    //      These are separate struct copies — without sync, weight changes
    //      have no effect on the forward pass.
    if (total_modified > 0) {
        for (uint32_t n = 0; n < network->num_neurons; n++) {
            neuron_t* neuron = &network->neurons[n];
            for (uint32_t s = 0; s < neuron->num_synapses; s++) {
                synapse_t* out = &neuron->synapses[s];
                if (out->target_id >= network->num_neurons) continue;
                neuron_t* target = &network->neurons[out->target_id];
                for (uint32_t k = 0; k < target->num_incoming; k++) {
                    if (target->incoming_synapses[k].source_neuron_id == n) {
                        target->incoming_synapses[k].weight = out->weight;
                        target->incoming_synapses[k].strength = out->strength;
                        break;
                    }
                }
            }
        }
    }

    LOG_INFO(LOG_MODULE, "Reward learning: modified %u synapses (reward=%.3f)",
             total_modified, reward);
    return total_modified;
}

uint32_t neural_network_apply_generalized_oja(neural_network_t network, uint32_t neuron_id,
                                              uint64_t timestamp)
{
    // Placeholder: Use standard Oja's rule for now
    // Can be extended with subspace tracking or PCA variants
    LOG_DEBUG(LOG_MODULE, "Generalized Oja not fully implemented, using standard Oja");
    return neural_network_apply_oja(network, neuron_id, timestamp);
}

uint32_t neural_network_update_plasticity(neural_network_t network, uint32_t neuron_id,
                                          uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons) {
        LOG_ERROR(LOG_MODULE, "Invalid network or neuron_id %u", neuron_id);
        return 0;
    }

    // Update meta-plasticity needs access to static helper in homeostasis module
    // For now, just return success
    LOG_DEBUG(LOG_MODULE, "Updated plasticity for neuron %u", neuron_id);
    return 1;
}
