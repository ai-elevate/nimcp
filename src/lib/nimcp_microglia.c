/**
 * @file nimcp_microglia.c
 * @brief Implementation of microglia glial cell module (synaptic surveillance & pruning)
 *
 * TDD STATUS: Stub implementation for RED phase
 * - All functions present but minimal implementation
 * - Basic memory management working
 * - Will implement full functionality in GREEN phase
 */

#include "nimcp_microglia.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_time.h"
#include <string.h>
#include <math.h>

// Default capacity for monitored synapses per microglia
#define MICROGLIA_DEFAULT_CAPACITY 1000

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

microglia_t* microglia_create(uint32_t id, float x, float y, float z, float surveillance_radius)
{
    if (surveillance_radius <= 0.0f) {
        return NULL;
    }

    microglia_t* mg = (microglia_t*) nimcp_malloc(sizeof(microglia_t));
    if (!mg) {
        return NULL;
    }

    memset(mg, 0, sizeof(microglia_t));

    mg->id = id;
    mg->position[0] = x;
    mg->position[1] = y;
    mg->position[2] = z;
    mg->surveillance_radius = surveillance_radius;

    // Allocate arrays for monitored synapses
    mg->max_monitored_synapses = MICROGLIA_DEFAULT_CAPACITY;
    mg->num_monitored_synapses = 0;

    mg->monitored_synapse_ids = (uint32_t*) nimcp_malloc(mg->max_monitored_synapses * sizeof(uint32_t));
    mg->synapse_activity_scores = (float*) nimcp_malloc(mg->max_monitored_synapses * sizeof(float));
    mg->last_activity_times = (uint64_t*) nimcp_malloc(mg->max_monitored_synapses * sizeof(uint64_t));

    if (!mg->monitored_synapse_ids || !mg->synapse_activity_scores || !mg->last_activity_times) {
        microglia_destroy(mg);
        return NULL;
    }

    // Initialize arrays
    for (uint32_t i = 0; i < mg->max_monitored_synapses; i++) {
        mg->monitored_synapse_ids[i] = UINT32_MAX;
        mg->synapse_activity_scores[i] = 0.0f;
        mg->last_activity_times[i] = 0;
    }

    // Set pruning parameters
    mg->pruning_threshold = NIMCP_MICROGLIA_PRUNING_THRESHOLD;
    mg->pruning_rate = NIMCP_MICROGLIA_PRUNING_RATE;
    mg->last_pruning_time = nimcp_time_monotonic_us();
    mg->total_synapses_pruned = 0;

    // Initialize lock
    nimcp_spinlock_init(&mg->lock);

    return mg;
}

void microglia_destroy(microglia_t* mg)
{
    if (!mg) return;

    if (mg->monitored_synapse_ids) {
        nimcp_free(mg->monitored_synapse_ids);
    }
    if (mg->synapse_activity_scores) {
        nimcp_free(mg->synapse_activity_scores);
    }
    if (mg->last_activity_times) {
        nimcp_free(mg->last_activity_times);
    }

    nimcp_free(mg);
}

// ============================================================================
// SYNAPSE MONITORING
// ============================================================================

nimcp_result_t microglia_monitor_synapse(microglia_t* mg, uint32_t synapse_id)
{
    if (!mg) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_spinlock_lock(&mg->lock);

    // Check for duplicate
    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->monitored_synapse_ids[i] == synapse_id) {
            nimcp_spinlock_unlock(&mg->lock);
            return NIMCP_SUCCESS; // Already monitored
        }
    }

    // Check capacity
    if (mg->num_monitored_synapses >= mg->max_monitored_synapses) {
        nimcp_spinlock_unlock(&mg->lock);
        return NIMCP_ERROR_INVALID_PARAM; // At capacity
    }

    // Add synapse
    uint32_t idx = mg->num_monitored_synapses;
    mg->monitored_synapse_ids[idx] = synapse_id;
    mg->synapse_activity_scores[idx] = 0.0f; // Start with 0 activity
    mg->last_activity_times[idx] = nimcp_time_monotonic_us();

    mg->num_monitored_synapses++;

    nimcp_spinlock_unlock(&mg->lock);
    return NIMCP_SUCCESS;
}

void microglia_track_synapse_activity(microglia_t* mg,
                                      uint32_t synapse_id,
                                      float activity,
                                      uint64_t timestamp)
{
    if (!mg) return;

    // Clamp activity to reasonable range
    if (activity < 0.0f) activity = 0.0f;
    if (activity > 10.0f) activity = 10.0f;

    nimcp_spinlock_lock(&mg->lock);

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->monitored_synapse_ids[i] == synapse_id) {
            // Update activity score
            // For first sample (score is 0), use activity directly
            // Otherwise use exponential moving average for smoothing
            if (mg->synapse_activity_scores[i] < 0.001f) {
                // First meaningful activity - set directly
                mg->synapse_activity_scores[i] = activity;
            } else {
                // Subsequent samples - use EMA for smoothing
                float alpha = 0.3f; // Smoothing factor (faster adaptation)
                mg->synapse_activity_scores[i] =
                    alpha * activity + (1.0f - alpha) * mg->synapse_activity_scores[i];
            }

            mg->last_activity_times[i] = timestamp;
            break;
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
}

void microglia_update_activity_scores(microglia_t* mg, uint64_t current_time)
{
    if (!mg) return;

    nimcp_spinlock_lock(&mg->lock);

    float tau_us = NIMCP_MICROGLIA_ACTIVITY_DECAY_TAU_S * 1000000.0f; // Convert to µs

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        uint64_t dt_us = current_time - mg->last_activity_times[i];
        float dt_s = (float)dt_us / 1000000.0f;

        // Exponential decay: score *= exp(-dt / tau)
        float decay_factor = expf(-dt_s * 1000000.0f / tau_us);
        mg->synapse_activity_scores[i] *= decay_factor;

        // Clamp to 0
        if (mg->synapse_activity_scores[i] < 0.001f) {
            mg->synapse_activity_scores[i] = 0.0f;
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
}

float microglia_get_synapse_activity_score(microglia_t* mg, uint32_t synapse_id)
{
    if (!mg) return 0.0f;

    nimcp_spinlock_lock(&mg->lock);

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->monitored_synapse_ids[i] == synapse_id) {
            float score = mg->synapse_activity_scores[i];
            nimcp_spinlock_unlock(&mg->lock);
            return score;
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
    return 0.0f; // Not found
}

// ============================================================================
// WEAK SYNAPSE IDENTIFICATION & PRUNING
// ============================================================================

uint32_t microglia_identify_weak_synapses(microglia_t* mg,
                                          uint32_t* weak_synapse_ids,
                                          uint32_t max_count)
{
    if (!mg || !weak_synapse_ids || max_count == 0) {
        return 0;
    }

    nimcp_spinlock_lock(&mg->lock);

    uint32_t num_weak = 0;

    for (uint32_t i = 0; i < mg->num_monitored_synapses && num_weak < max_count; i++) {
        if (mg->synapse_activity_scores[i] < mg->pruning_threshold) {
            weak_synapse_ids[num_weak] = mg->monitored_synapse_ids[i];
            num_weak++;
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
    return num_weak;
}

uint32_t microglia_prune_weak_synapses(microglia_t* mg)
{
    if (!mg || mg->num_monitored_synapses == 0) {
        return 0;
    }

    nimcp_spinlock_lock(&mg->lock);

    uint32_t num_pruned = 0;
    uint32_t max_prune = (uint32_t)mg->pruning_rate;

    // Identify and remove weak synapses
    uint32_t write_idx = 0;
    for (uint32_t read_idx = 0; read_idx < mg->num_monitored_synapses; read_idx++) {
        bool should_prune = (mg->synapse_activity_scores[read_idx] < mg->pruning_threshold)
                            && (num_pruned < max_prune);

        if (should_prune) {
            num_pruned++;
            // Don't copy this entry (effectively removing it)
        } else {
            // Keep this synapse - copy to write position
            if (write_idx != read_idx) {
                mg->monitored_synapse_ids[write_idx] = mg->monitored_synapse_ids[read_idx];
                mg->synapse_activity_scores[write_idx] = mg->synapse_activity_scores[read_idx];
                mg->last_activity_times[write_idx] = mg->last_activity_times[read_idx];
            }
            write_idx++;
        }
    }

    // Update count
    mg->num_monitored_synapses = write_idx;
    mg->total_synapses_pruned += num_pruned;
    mg->last_pruning_time = nimcp_time_monotonic_us();

    nimcp_spinlock_unlock(&mg->lock);
    return num_pruned;
}

// ============================================================================
// NETWORK MANAGEMENT
// ============================================================================

microglia_network_t* microglia_network_create(uint32_t capacity)
{
    if (capacity == 0) return NULL;

    microglia_network_t* network =
        (microglia_network_t*) nimcp_malloc(sizeof(microglia_network_t));
    if (!network) return NULL;

    memset(network, 0, sizeof(microglia_network_t));

    network->capacity = capacity;
    network->num_microglia = 0;

    network->microglia =
        (microglia_t**) nimcp_malloc(capacity * sizeof(microglia_t*));
    if (!network->microglia) {
        nimcp_free(network);
        return NULL;
    }

    for (uint32_t i = 0; i < capacity; i++) {
        network->microglia[i] = NULL;
    }

    // Global parameters
    network->global_pruning_threshold = NIMCP_MICROGLIA_PRUNING_THRESHOLD;
    network->min_activity_window_ms = NIMCP_MICROGLIA_MIN_ACTIVITY_WINDOW_MS;

    nimcp_mutex_init(&network->lock, NULL);

    return network;
}

void microglia_network_destroy(microglia_network_t* network)
{
    if (!network) return;

    // Destroy all microglia
    for (uint32_t i = 0; i < network->num_microglia; i++) {
        if (network->microglia[i]) {
            microglia_destroy(network->microglia[i]);
        }
    }

    if (network->microglia) {
        nimcp_free(network->microglia);
    }

    nimcp_mutex_destroy(&network->lock);
    nimcp_free(network);
}

nimcp_result_t microglia_network_add(microglia_network_t* network, microglia_t* mg)
{
    if (!network || !mg) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&network->lock);

    if (network->num_microglia >= network->capacity) {
        nimcp_mutex_unlock(&network->lock);
        return NIMCP_ERROR_INVALID_PARAM; // At capacity
    }

    network->microglia[network->num_microglia] = mg;
    network->num_microglia++;

    nimcp_mutex_unlock(&network->lock);
    return NIMCP_SUCCESS;
}

void microglia_network_step(microglia_network_t* network, uint64_t current_time)
{
    if (!network) return;

    nimcp_mutex_lock(&network->lock);

    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg = network->microglia[i];
        if (mg) {
            // Update activity scores (decay over time)
            microglia_update_activity_scores(mg, current_time);

            // Prune weak synapses
            microglia_prune_weak_synapses(mg);
        }
    }

    nimcp_mutex_unlock(&network->lock);
}

microglia_t* microglia_network_find_by_synapse(microglia_network_t* network,
                                                uint32_t synapse_id)
{
    if (!network) return NULL;

    nimcp_mutex_lock(&network->lock);

    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg = network->microglia[i];
        if (!mg) continue;

        for (uint32_t j = 0; j < mg->num_monitored_synapses; j++) {
            if (mg->monitored_synapse_ids[j] == synapse_id) {
                nimcp_mutex_unlock(&network->lock);
                return mg;
            }
        }
    }

    nimcp_mutex_unlock(&network->lock);
    return NULL;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

uint32_t microglia_get_total_pruned(microglia_t* mg)
{
    if (!mg) return 0;
    return mg->total_synapses_pruned;
}
