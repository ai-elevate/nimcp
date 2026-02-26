#include <stddef.h>  /* for NULL */
#include <float.h>   /* for FLT_MAX */
//=============================================================================
// nimcp_neuralnet_learning.c - Learning Rules and Synaptic Plasticity
//=============================================================================
// NOTE: NEURALNET_LEARNING_SEPARATE is defined via CMake to guard duplicates
//       in nimcp_neuralnet.c when this file is compiled.

#include "core/neuralnet/nimcp_neuralnet_learning.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
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
#define ACTIVITY_THRESHOLD 1e-5f

/* Single authoritative definition of neural_network_struct */
#include "core/neuralnet/nimcp_neuralnet_internal.h"

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
        return params->positive_factor * expf(-dt / time_window);
    } else {
        // Post before pre - depression
        return -params->negative_factor * expf(dt / time_window);
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

static void update_synaptic_traces_sparse(neuron_t* neuron,
                                          synapse_metadata_pool_t meta_pool,
                                          uint64_t timestamp)
{
    if (!neuron)
        return;

    float dt = (timestamp > neuron->last_update) ?
               (float)(timestamp - neuron->last_update) : 1.0f;

    float decay = expf(-TRACE_DECAY_RATE * dt);

    /* Iterate outgoing synapses via sparse storage API */
    sparse_synapse_iterator_t it;
    sparse_synapse_iterator_init(&it, &neuron->outgoing);
    synapse_handle_t* handle;
    while ((handle = sparse_synapse_iterator_next(&it)) != NULL) {
        /* Update trace on metadata if available */
        if (handle->metadata_index != SPARSE_SYNAPSE_NO_METADATA && meta_pool) {
            synapse_t* meta = synapse_metadata_pool_get(meta_pool, handle->metadata_index);
            if (meta) {
                meta->trace *= decay;
            }
        }
    }
}

static void normalize_synaptic_weights_sparse(neuron_t* neuron)
{
    if (!neuron) return;
    uint32_t syn_count = NEURON_OUT_COUNT(neuron);
    if (syn_count == 0)
        return;

    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < syn_count; i++) {
        synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, i);
        if (!h) continue;
        float w = h->weight;
        sum_sq += w * w;
    }

    if (sum_sq < NORM_THRESHOLD)
        return;

    float norm = sqrtf(sum_sq);
    neuron->weight_norm = norm;

    float target_norm = neuron->oja_params.target_norm;
    if (norm > target_norm * 1.5f) {
        float scale = target_norm / norm;
        for (uint32_t i = 0; i < syn_count; i++) {
            synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, i);
            if (!h) continue;
            h->weight *= scale;
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
    normalize_synaptic_weights_sparse(neuron);

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
    update_synaptic_traces_sparse(neuron, network->synapse_metadata_pool, timestamp);
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
    uint32_t syn_count = NEURON_OUT_COUNT(neuron);

    if (syn_count == 0) {
        if (mean) *mean = 0.0f;
        if (std_dev) *std_dev = 0.0f;
        if (min_weight) *min_weight = 0.0f;
        if (max_weight) *max_weight = 0.0f;
        return;
    }

    float sum = 0.0f;
    float min_w = FLT_MAX;
    float max_w = -FLT_MAX;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i < syn_count; i++) {
        synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, i);
        if (!h) continue;
        float w = h->weight;
        sum += w;
        if (w < min_w) min_w = w;
        if (w > max_w) max_w = w;
        valid_count++;
    }

    /* Guard: all handles were NULL despite syn_count > 0 */
    if (valid_count == 0) {
        if (mean) *mean = 0.0f;
        if (std_dev) *std_dev = 0.0f;
        if (min_weight) *min_weight = 0.0f;
        if (max_weight) *max_weight = 0.0f;
        return;
    }

    float avg = sum / (float)valid_count;

    float variance = 0.0f;
    for (uint32_t i = 0; i < syn_count; i++) {
        synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, i);
        if (!h) continue;
        float diff = h->weight - avg;
        variance += diff * diff;
    }
    variance /= (float)valid_count;

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
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neural_network_apply_oja: network is NULL");
        return 0;
    }
    if (neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OUT_OF_RANGE, "neural_network_apply_oja: neuron_id %u out of range", neuron_id);
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

    // Calculate weight updates using Oja's rule via sparse synapse API
    uint32_t syn_count = NEURON_OUT_COUNT(neuron);
    for (uint32_t i = 0; i < syn_count; i++) {
        synapse_handle_t* handle = NEURON_OUT_HANDLE(neuron, i);
        if (!handle) continue;

        uint32_t target_id = handle->target_neuron_id;
        if (target_id >= network->num_neurons) continue;

        // Get post-synaptic average activity (target of synapse)
        float y = network->neurons[target_id].avg_activity;

        // Compute weight update using Oja's rule: Δw = α(y*x - y²*w)
        float delta_w = compute_oja_weight_update(x, y, handle->weight, &neuron->oja_params);

        // Get meta-plasticity from metadata if available
        float meta_plasticity = 1.0f;
        synapse_t* meta = NULL;
        if (handle->metadata_index != SPARSE_SYNAPSE_NO_METADATA &&
            network->synapse_metadata_pool) {
            meta = synapse_metadata_pool_get(network->synapse_metadata_pool,
                                             handle->metadata_index);
            if (meta) {
                meta_plasticity = meta->meta_plasticity;
            }
        }

        // Apply weight update with meta-plasticity
        float new_weight = handle->weight + delta_w * meta_plasticity;

        // Guard against NaN/Inf from numerical instability
        if (!isfinite(new_weight)) new_weight = 0.0f;

        // Apply weight constraints
        new_weight = fmaxf(network->config.min_weight,
                          fminf(network->config.max_weight, new_weight));

        // Update weight if change is significant
        if (fabsf(new_weight - handle->weight) > WEIGHT_UPDATE_THRESHOLD) {
            handle->weight = new_weight;
            modified++;
        }

        // Update synaptic strength
        float new_strength = fminf(handle->strength * (1.0f + delta_w), MAX_SYNAPTIC_STRENGTH);
        if (!isfinite(new_strength)) new_strength = 1.0f;
        handle->strength = new_strength;
        if (meta) {
            meta->weight = handle->weight;
            meta->strength = new_strength;
        }
    }

    // Normalize weights periodically
    if (modified > 0 && (timestamp - neuron->last_update) > NORMALIZATION_INTERVAL) {
        normalize_synaptic_weights_sparse(neuron);
    }

    LOG_DEBUG(LOG_MODULE, "Applied Oja's rule to neuron %u, modified %u synapses",
              neuron_id, modified);
    return modified;
}

uint32_t neural_network_apply_stdp(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neural_network_apply_stdp: network is NULL");
        return 0;
    }
    if (neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OUT_OF_RANGE, "neural_network_apply_stdp: neuron_id %u out of range", neuron_id);
        return 0;
    }

    neuron_t* pre_neuron = &network->neurons[neuron_id];
    uint32_t modified = 0;

    // Iterate over all outgoing synapses from this neuron via sparse API
    uint32_t syn_count = NEURON_OUT_COUNT(pre_neuron);
    for (uint32_t i = 0; i < syn_count; i++) {
        synapse_handle_t* handle = NEURON_OUT_HANDLE(pre_neuron, i);
        if (!handle) continue;

        uint32_t target_id = handle->target_neuron_id;
        if (target_id >= network->num_neurons) continue;

        // Get postsynaptic neuron (target of this synapse)
        const neuron_t* post_neuron = &network->neurons[target_id];

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

        // Get trace and meta-plasticity from metadata if available
        float trace_val = 1.0f;
        float meta_plasticity = 1.0f;
        synapse_t* meta = NULL;
        if (handle->metadata_index != SPARSE_SYNAPSE_NO_METADATA &&
            network->synapse_metadata_pool) {
            meta = synapse_metadata_pool_get(network->synapse_metadata_pool,
                                             handle->metadata_index);
            if (meta) {
                trace_val = meta->trace;
                meta_plasticity = meta->meta_plasticity;
            }
        }

        // Scale STDP by learning rate, trace, and meta-plasticity
        float delta_w = pre_neuron->stdp_params.learning_rate * stdp_factor * trace_val;

        // Apply weight update with meta-plasticity modulation
        float new_weight = handle->weight + delta_w * meta_plasticity;

        // Guard against NaN/Inf from numerical instability
        if (!isfinite(new_weight)) new_weight = 0.0f;

        // Clamp weight to configured bounds
        new_weight = fmaxf(network->config.min_weight,
                          fminf(network->config.max_weight, new_weight));

        // Update weight if change is significant
        if (fabsf(new_weight - handle->weight) > WEIGHT_UPDATE_THRESHOLD) {
            handle->weight = new_weight;
            if (meta) meta->weight = new_weight;
            modified++;
        }

        // Apply BCM homeostatic plasticity after STDP (if enabled)
        if (meta && meta->enable_bcm && meta->bcm) {
            bcm_params_t bcm_params = bcm_params_cortical();
            float dt_sec = (timestamp > meta->last_active) ?
                (float)(timestamp - meta->last_active) / 1000.0f : 0.001f;
            float pre_activity = trace_val;
            float post_activity = post_neuron->state;

            bcm_apply_rule(meta->bcm, pre_activity, post_activity, dt_sec, &bcm_params);

            if (fabsf(meta->bcm->weight - handle->weight) > WEIGHT_UPDATE_THRESHOLD) {
                handle->weight = meta->bcm->weight;
                meta->weight = meta->bcm->weight;
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
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neural_network_apply_lateral_inhibition: network is NULL");
        return 0;
    }
    if (output_count == 0) return 0;
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
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neural_network_apply_reward_learning: network is NULL");
        return 0;
    }
    if (reward < 0.0f || reward > 1.0f || learning_rate <= 0.0f) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "neural_network_apply_reward_learning: invalid reward=%.3f or lr=%.3f", reward, learning_rate);
        return 0;
    }

    uint32_t total_modified = 0;

    // PRE-PASS: Set outgoing synapse traces from current neuron states
    // WHY: The forward pass (neural_network_forward) sets neuron->state via
    //      incoming synapses but never updates outgoing synapse traces.
    //      Eligibility learning needs syn->trace > 0.1 to trigger updates.
    //      Set each neuron's outgoing synapse traces to |neuron->state| so
    //      eligibility-based credit assignment knows which synapses were active.
    for (uint32_t n = 0; n < network->num_neurons; n++) {
        float activity = fabsf(network->neurons[n].state);
        neuron_t* neuron = &network->neurons[n];
        uint32_t out_count = NEURON_OUT_COUNT(neuron);
        for (uint32_t s = 0; s < out_count; s++) {
            synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, s);
            if (!h) continue;
            if (h->metadata_index != SPARSE_SYNAPSE_NO_METADATA &&
                network->synapse_metadata_pool) {
                synapse_t* meta = synapse_metadata_pool_get(
                    network->synapse_metadata_pool, h->metadata_index);
                if (meta) {
                    meta->trace = activity;
                }
            }
        }
    }

    // Iterate over all neurons in the network (use num_neurons to include dynamically-added)
    for (uint32_t neuron_id = 0; neuron_id < network->num_neurons; neuron_id++) {
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
        uint32_t syn_count = NEURON_OUT_COUNT(neuron);
        for (uint32_t syn_idx = 0; syn_idx < syn_count; syn_idx++) {
            synapse_handle_t* handle = NEURON_OUT_HANDLE(neuron, syn_idx);
            if (!handle) continue;

            // Get metadata for eligibility/BCM access
            synapse_t* syn = NULL;
            if (handle->metadata_index != SPARSE_SYNAPSE_NO_METADATA &&
                network->synapse_metadata_pool) {
                syn = synapse_metadata_pool_get(network->synapse_metadata_pool,
                                                handle->metadata_index);
            }

            // Update eligibility traces (requires metadata)
            if (syn && syn->enable_eligibility && syn->eligibility) {
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
                float old_weight = handle->weight;
                syn->weight = handle->weight;  // sync before eligibility modifies it
                eligibility_apply_reward(syn, syn->eligibility, &elig_config, reward, dopamine);

                // Guard against NaN/Inf from numerical instability
                if (!isfinite(syn->weight)) syn->weight = old_weight;

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
                    handle->weight = syn->weight;  // sync back to handle
                    total_modified++;
                }
            }

            // Apply BCM homeostatic plasticity (requires metadata)
            if (syn && syn->enable_bcm && syn->bcm) {
                uint32_t target_id = handle->target_neuron_id;
                if (target_id < network->num_neurons) {
                    bcm_params_t bcm_params = bcm_params_cortical();
                    const neuron_t* post_neuron = &network->neurons[target_id];

                    bcm_apply_rule(syn->bcm, neuron->state, post_neuron->state, 1.0f, &bcm_params);

                    if (fabsf(syn->bcm->weight - handle->weight) > WEIGHT_UPDATE_THRESHOLD) {
                        handle->weight = syn->bcm->weight;
                        syn->weight = syn->bcm->weight;
                        total_modified++;
                    }
                }
            }
        }
    }

    // POST-PASS: Sync outgoing synapse weights to incoming synapses
    // WHY: Learning modifies outgoing synapse handle weights, but the forward
    //      pass reads from incoming synapse handles. With sparse storage,
    //      outgoing and incoming handles are separate — must sync weights.
    //      Peer-index cross-references enable efficient O(1) sync per synapse.
    if (total_modified > 0) {
        for (uint32_t n = 0; n < network->num_neurons; n++) {
            neuron_t* neuron = &network->neurons[n];
            uint32_t out_count = NEURON_OUT_COUNT(neuron);
            for (uint32_t s = 0; s < out_count; s++) {
                synapse_handle_t* out_h = NEURON_OUT_HANDLE(neuron, s);
                if (!out_h) continue;
                uint32_t target_id = out_h->target_neuron_id;
                if (target_id >= network->num_neurons) continue;

                // Use peer_index for O(1) sync if available
                if (out_h->peer_index != SPARSE_SYNAPSE_NO_PEER) {
                    neuron_t* target = &network->neurons[target_id];
                    synapse_handle_t* in_h = NEURON_IN_HANDLE(target, out_h->peer_index);
                    if (in_h) {
                        in_h->weight = out_h->weight;
                        in_h->strength = out_h->strength;
                    }
                } else {
                    // Fallback: linear scan of target's incoming synapses
                    neuron_t* target = &network->neurons[target_id];
                    uint32_t in_count = NEURON_IN_COUNT(target);
                    for (uint32_t k = 0; k < in_count; k++) {
                        synapse_handle_t* in_h = NEURON_IN_HANDLE(target, k);
                        if (in_h && in_h->target_neuron_id == n) {
                            in_h->weight = out_h->weight;
                            in_h->strength = out_h->strength;
                            break;
                        }
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
    if (!network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neural_network_update_plasticity: network is NULL");
        return 0;
    }
    if (neuron_id >= network->num_neurons) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OUT_OF_RANGE, "neural_network_update_plasticity: neuron_id %u out of range", neuron_id);
        return 0;
    }

    // Update meta-plasticity needs access to static helper in homeostasis module
    // For now, just return success
    LOG_DEBUG(LOG_MODULE, "Updated plasticity for neuron %u", neuron_id);
    return 1;
}
