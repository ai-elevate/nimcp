/**
 * @file nimcp_astrocytes.c
 * @brief Implementation of biological astrocyte glial cells
 *
 * STATUS: Initial TDD stubs - tests will FAIL (RED phase)
 * NEXT: Implement full functionality to make tests pass (GREEN phase)
 */

#include "nimcp_astrocytes.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_time.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Creation and Destruction
//=============================================================================

astrocyte_t* astrocyte_create(uint32_t id, float x, float y, float z, float coverage_radius)
{
    if (coverage_radius < 0.0f) {
        return NULL;
    }

    astrocyte_t* astro = (astrocyte_t*) nimcp_malloc(sizeof(astrocyte_t));
    if (!astro) {
        return NULL;
    }

    memset(astro, 0, sizeof(astrocyte_t));

    // Identity
    astro->id = id;

    // Calcium dynamics - initialize to resting baseline
    astro->calcium_concentration = ASTROCYTE_BASELINE_CALCIUM_UM;
    astro->calcium_baseline = ASTROCYTE_BASELINE_CALCIUM_UM;
    astro->ip3_concentration = 0.0f;
    astro->last_calcium_spike = 0;

    // Neurotransmitter pools - initialize to full
    astro->glutamate_pool = 1.0f;
    astro->d_serine_pool = 1.0f;
    astro->atp_level = 1.0f;

    // Spatial location
    astro->position[0] = x;
    astro->position[1] = y;
    astro->position[2] = z;
    astro->coverage_radius = coverage_radius;

    // Synaptic coverage - initially empty
    astro->num_covered_synapses = 0;
    astro->covered_synapse_ids = NULL;
    astro->synapse_calcium_levels = NULL;

    // Gap junction coupling - initially empty
    astro->num_coupled_astrocytes = 0;
    astro->coupled_astrocyte_ids = NULL;
    astro->coupling_strengths = NULL;

    // Homeostatic regulation
    astro->target_activity_level = 0.3f; // Default target
    astro->scaling_factor = 1.0f;

    // Thread safety
    nimcp_spinlock_init(&astro->lock);

    return astro;
}

void astrocyte_destroy(astrocyte_t* astro)
{
    if (!astro) {
        return;
    }

    // Free synapse arrays
    if (astro->covered_synapse_ids) {
        nimcp_free(astro->covered_synapse_ids);
    }
    if (astro->synapse_calcium_levels) {
        nimcp_free(astro->synapse_calcium_levels);
    }

    // Free coupling arrays
    if (astro->coupled_astrocyte_ids) {
        nimcp_free(astro->coupled_astrocyte_ids);
    }
    if (astro->coupling_strengths) {
        nimcp_free(astro->coupling_strengths);
    }

    // Destroy spinlock
    nimcp_spinlock_destroy(&astro->lock);

    // Free astrocyte itself
    nimcp_free(astro);
}

//=============================================================================
// Calcium Dynamics - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_update_calcium(astrocyte_t* astro, float dt, float external_stimulus)
{
    if (!astro || dt < 0.0f) {
        return;
    }

    // TODO: Implement full calcium dynamics (Li-Rinzel model or simplified)
    // For now, simple decay back to baseline + stimulus integration

    nimcp_spinlock_lock(&astro->lock);

    // Simple decay towards baseline
    float tau_decay = 0.1f; // 100ms time constant
    float decay = (astro->calcium_concentration - astro->calcium_baseline) / tau_decay;

    // Add external stimulus
    float stimulus_gain = 0.5f;
    astro->calcium_concentration += dt * (stimulus_gain * external_stimulus - decay);

    // Clamp to reasonable range
    astro->calcium_concentration = fmaxf(0.0f, fminf(50.0f, astro->calcium_concentration));

    // Simple IP3 production for strong stimulation
    if (external_stimulus > 10.0f) {
        astro->ip3_concentration += dt * external_stimulus * 0.1f;
        astro->ip3_concentration = fminf(5.0f, astro->ip3_concentration);
    }

    // Decay IP3
    astro->ip3_concentration *= expf(-dt / 0.2f); // 200ms time constant

    // Detect calcium spike
    if (astro->calcium_concentration > astro->calcium_baseline * 3.0f) {
        astro->last_calcium_spike = nimcp_time_monotonic_us();
    }

    nimcp_spinlock_unlock(&astro->lock);
}

void astrocyte_propagate_calcium_wave(astrocyte_t* astro,
                                      astrocyte_network_t* network,
                                      float dt)
{
    if (!astro || !network) {
        return;
    }

    // Only propagate if calcium is above threshold
    if (astro->calcium_concentration < network->calcium_threshold_um) {
        return;
    }

    // Propagate calcium to coupled neighbors via gap junctions
    nimcp_spinlock_lock(&astro->lock);

    for (uint32_t i = 0; i < astro->num_coupled_astrocytes; i++) {
        uint32_t neighbor_id = astro->coupled_astrocyte_ids[i];
        float coupling_strength = astro->coupling_strengths[i];

        // Find neighbor astrocyte in network
        astrocyte_t* neighbor = NULL;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (network->astrocytes[j]->id == neighbor_id) {
                neighbor = network->astrocytes[j];
                break;
            }
        }

        if (neighbor) {
            // Calcium diffusion through gap junction
            float ca_diff = astro->calcium_concentration - neighbor->calcium_concentration;
            float flux = coupling_strength * ca_diff * network->coupling_decay_rate * dt;

            // Update neighbor calcium (with their lock)
            nimcp_spinlock_lock(&neighbor->lock);
            neighbor->calcium_concentration += flux;
            neighbor->calcium_concentration = fmaxf(0.0f, fminf(50.0f, neighbor->calcium_concentration));
            nimcp_spinlock_unlock(&neighbor->lock);
        }
    }

    nimcp_spinlock_unlock(&astro->lock);
}

//=============================================================================
// Neurotransmitter Release - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_glutamate_release(astrocyte_t* astro, uint32_t synapse_idx)
{
    if (!astro || synapse_idx >= astro->num_covered_synapses) {
        return 0.0f;
    }

    // Calcium-dependent glutamate release with pool depletion
    float ca_threshold = astro->calcium_baseline * 2.0f;
    if (astro->calcium_concentration > ca_threshold) {
        float excess = astro->calcium_concentration - ca_threshold;
        float release_fraction = fminf(1.0f, excess / 5.0f);

        // Release glutamate from pool (depletes pool)
        // Higher release rate for biological accuracy
        float release_rate = 0.15f; // 15% of available pool × release_fraction
        float release_amount = release_fraction * astro->glutamate_pool * release_rate;
        astro->glutamate_pool -= release_amount;
        astro->glutamate_pool = fmaxf(0.0f, astro->glutamate_pool); // Clamp to [0, 1]

        return release_amount;
    }

    return 0.0f;
}

float astrocyte_compute_d_serine_release(astrocyte_t* astro, uint32_t synapse_idx)
{
    if (!astro || synapse_idx >= astro->num_covered_synapses) {
        return 0.0f;
    }

    // TODO: Implement full D-serine release model
    // For now, similar to glutamate but lower threshold

    float ca_threshold = astro->calcium_baseline * 1.5f;
    if (astro->calcium_concentration > ca_threshold) {
        float excess = astro->calcium_concentration - ca_threshold;
        return fminf(1.0f, excess / 3.0f) * astro->d_serine_pool;
    }

    return 0.0f;
}

//=============================================================================
// Synaptic Modulation - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_modulate_synapse_strength(astrocyte_t* astro,
                                         synapse_t* synapse,
                                         uint32_t synapse_idx)
{
    if (!astro || !synapse || synapse_idx >= astro->num_covered_synapses) {
        return;
    }

    // TODO: Implement full modulation based on glutamate/D-serine
    // For now, simple multiplicative enhancement

    float glutamate = astrocyte_compute_glutamate_release(astro, synapse_idx);
    float modulation = 1.0f + 0.5f * glutamate; // Up to 50% enhancement

    synapse->strength *= modulation;
}

nimcp_result_t astrocyte_assign_synapse(astrocyte_t* astro, uint32_t synapse_id)
{
    if (!astro) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Reallocate arrays
    uint32_t new_count = astro->num_covered_synapses + 1;

    uint32_t* new_ids = (uint32_t*) nimcp_realloc(astro->covered_synapse_ids,
                                                    new_count * sizeof(uint32_t));
    if (!new_ids) {
        return NIMCP_ERROR_MEMORY;
    }
    astro->covered_synapse_ids = new_ids;

    float* new_ca = (float*) nimcp_realloc(astro->synapse_calcium_levels,
                                            new_count * sizeof(float));
    if (!new_ca) {
        return NIMCP_ERROR_MEMORY;
    }
    astro->synapse_calcium_levels = new_ca;

    // Add synapse
    astro->covered_synapse_ids[astro->num_covered_synapses] = synapse_id;
    astro->synapse_calcium_levels[astro->num_covered_synapses] = astro->calcium_baseline;
    astro->num_covered_synapses = new_count;

    return NIMCP_SUCCESS;
}

//=============================================================================
// Homeostatic Plasticity - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_synaptic_scaling(astrocyte_t* astro, neural_network_t* network)
{
    if (!astro) {
        return 1.0f;
    }

    // TODO: Implement full homeostatic scaling based on activity estimation
    // For now, simple calcium-based scaling

    float current_activity = astro->calcium_concentration / 5.0f; // Normalize
    float target = astro->target_activity_level;
    float error = target - current_activity;

    float gain = 0.1f;
    float scaling = 1.0f + gain * error;

    // Clamp to reasonable range
    return fmaxf(0.5f, fminf(2.0f, scaling));
}

//=============================================================================
// BCM Threshold Modulation - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_bcm_threshold_shift(astrocyte_t* astro, float default_threshold)
{
    if (!astro) {
        return 0.0f;
    }

    // TODO: Implement full BCM modulation model
    // For now, simple calcium-dependent shift

    float calcium_excess = astro->calcium_concentration - astro->calcium_baseline;
    float shift_gain = 0.05f;

    return shift_gain * calcium_excess;
}

//=============================================================================
// Metabolic Support - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_update_atp_level(astrocyte_t* astro, float neural_activity, float dt)
{
    if (!astro || dt < 0.0f) {
        return;
    }

    // TODO: Implement full ATP production/consumption model
    // For now, simple dynamics

    float production_rate = 0.5f; // Basal production
    float consumption_rate = 0.1f * neural_activity; // Activity-dependent

    float d_atp = (production_rate - consumption_rate) * dt;
    astro->atp_level += d_atp;

    // Clamp to [0, 1]
    astro->atp_level = fmaxf(0.0f, fminf(1.0f, astro->atp_level));
}

//=============================================================================
// Astrocyte Network Management - STUB IMPLEMENTATION
//=============================================================================

astrocyte_network_t* astrocyte_network_create(uint32_t capacity)
{
    astrocyte_network_t* network = (astrocyte_network_t*) nimcp_malloc(sizeof(astrocyte_network_t));
    if (!network) {
        return NULL;
    }

    memset(network, 0, sizeof(astrocyte_network_t));

    network->capacity = capacity;
    network->num_astrocytes = 0;
    network->astrocytes = (astrocyte_t**) nimcp_malloc(capacity * sizeof(astrocyte_t*));
    if (!network->astrocytes) {
        nimcp_free(network);
        return NULL;
    }

    // Global parameters
    network->calcium_threshold_um = ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM;
    network->coupling_decay_rate = 2.0f; // Stronger coupling for faster wave propagation
    network->coupling_radius_um = ASTROCYTE_COUPLING_RADIUS_UM;

    // Spatial index (NULL for now)
    network->spatial_index = NULL;

    nimcp_mutex_init(&network->lock, NULL);

    return network;
}

void astrocyte_network_destroy(astrocyte_network_t* network)
{
    if (!network) {
        return;
    }

    // Destroy all astrocytes
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_destroy(network->astrocytes[i]);
    }

    // Free arrays
    if (network->astrocytes) {
        nimcp_free(network->astrocytes);
    }

    // TODO: Destroy spatial index when implemented

    nimcp_mutex_destroy(&network->lock);
    nimcp_free(network);
}

nimcp_result_t astrocyte_network_add(astrocyte_network_t* network, astrocyte_t* astro)
{
    if (!network || !astro) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (network->num_astrocytes >= network->capacity) {
        // Resize
        uint32_t new_capacity = network->capacity * 2;
        astrocyte_t** new_array = (astrocyte_t**) nimcp_realloc(network->astrocytes,
                                                                 new_capacity * sizeof(astrocyte_t*));
        if (!new_array) {
            return NIMCP_ERROR_MEMORY;
        }
        network->astrocytes = new_array;
        network->capacity = new_capacity;
    }

    network->astrocytes[network->num_astrocytes++] = astro;
    return NIMCP_SUCCESS;
}

nimcp_result_t astrocyte_network_build_spatial_index(astrocyte_network_t* network)
{
    if (!network) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // TODO: Build KD-tree for spatial indexing
    // For now, stub returns success
    return NIMCP_SUCCESS;
}

astrocyte_t* astrocyte_network_find_nearest(astrocyte_network_t* network, const float point[3])
{
    if (!network || !point || network->num_astrocytes == 0) {
        return NULL;
    }

    // TODO: Use KD-tree for O(log N) lookup
    // For now, linear search O(N)

    astrocyte_t* nearest = network->astrocytes[0];
    float min_dist_sq = INFINITY;

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        float dx = astro->position[0] - point[0];
        float dy = astro->position[1] - point[1];
        float dz = astro->position[2] - point[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest = astro;
        }
    }

    return nearest;
}

nimcp_result_t astrocyte_network_establish_coupling(astrocyte_network_t* network)
{
    if (!network) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // For each astrocyte, find neighbors within coupling radius
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];

        // Count neighbors first
        uint32_t neighbor_count = 0;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (i == j) continue;

            astrocyte_t* neighbor = network->astrocytes[j];
            float dx = astro->position[0] - neighbor->position[0];
            float dy = astro->position[1] - neighbor->position[1];
            float dz = astro->position[2] - neighbor->position[2];
            float distance = sqrtf(dx*dx + dy*dy + dz*dz);

            if (distance <= network->coupling_radius_um) {
                neighbor_count++;
            }
        }

        if (neighbor_count == 0) {
            continue;
        }

        // Allocate arrays for neighbors
        astro->coupled_astrocyte_ids = (uint32_t*) nimcp_malloc(neighbor_count * sizeof(uint32_t));
        astro->coupling_strengths = (float*) nimcp_malloc(neighbor_count * sizeof(float));

        if (!astro->coupled_astrocyte_ids || !astro->coupling_strengths) {
            return NIMCP_ERROR_MEMORY;
        }

        // Fill arrays
        uint32_t idx = 0;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (i == j) continue;

            astrocyte_t* neighbor = network->astrocytes[j];
            float dx = astro->position[0] - neighbor->position[0];
            float dy = astro->position[1] - neighbor->position[1];
            float dz = astro->position[2] - neighbor->position[2];
            float distance = sqrtf(dx*dx + dy*dy + dz*dz);

            if (distance <= network->coupling_radius_um) {
                astro->coupled_astrocyte_ids[idx] = neighbor->id;

                // Coupling strength decreases with distance
                // Use max(0.1, ...) to ensure minimum coupling even at boundary
                float normalized_dist = distance / network->coupling_radius_um;
                float strength = fmaxf(0.1f, 1.0f - normalized_dist);
                astro->coupling_strengths[idx] = strength;

                idx++;
            }
        }

        astro->num_coupled_astrocytes = neighbor_count;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t astrocyte_network_assign_synapses(astrocyte_network_t* network,
                                                 neural_network_t* nn)
{
    if (!network || !nn) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // TODO: Implement full synapse assignment based on spatial proximity
    // For now, stub returns success
    return NIMCP_SUCCESS;
}

void astrocyte_network_step(astrocyte_network_t* network, float dt)
{
    if (!network || dt <= 0.0f) {
        return;
    }

    // TODO: Implement full network dynamics
    // For now, simple update of all astrocytes

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];

        // Update calcium
        astrocyte_update_calcium(astro, dt, 0.0f);

        // Propagate waves
        astrocyte_propagate_calcium_wave(astro, network, dt);

        // Update ATP
        astrocyte_update_atp_level(astro, 1.0f, dt);
    }
}

//=============================================================================
// Utility Functions
//=============================================================================

void astrocyte_network_get_stats(astrocyte_network_t* network,
                                 float* avg_calcium,
                                 float* max_calcium,
                                 float* avg_glutamate)
{
    if (!network) {
        if (avg_calcium) *avg_calcium = 0.0f;
        if (max_calcium) *max_calcium = 0.0f;
        if (avg_glutamate) *avg_glutamate = 0.0f;
        return;
    }

    float sum_ca = 0.0f;
    float sum_glu = 0.0f;
    float max_ca = 0.0f;

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        sum_ca += astro->calcium_concentration;
        sum_glu += astro->glutamate_pool;
        if (astro->calcium_concentration > max_ca) {
            max_ca = astro->calcium_concentration;
        }
    }

    if (avg_calcium) *avg_calcium = sum_ca / network->num_astrocytes;
    if (max_calcium) *max_calcium = max_ca;
    if (avg_glutamate) *avg_glutamate = sum_glu / network->num_astrocytes;
}
