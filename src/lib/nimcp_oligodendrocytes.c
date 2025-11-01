/**
 * @file nimcp_oligodendrocytes.c
 * @brief Implementation of oligodendrocyte glial cell module (myelination & conduction velocity)
 *
 * TDD STATUS: Stub implementation for RED phase
 * - All functions present but minimal implementation
 * - Basic memory management working
 * - Will implement full functionality in GREEN phase
 */

#include "nimcp_oligodendrocytes.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_time.h"
#include <string.h>
#include <math.h>

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

oligodendrocyte_t* oligodendrocyte_create(uint32_t id, uint32_t max_axons)
{
    if (max_axons == 0 || max_axons > NIMCP_OLIGO_MAX_AXONS) {
        return NULL;
    }

    oligodendrocyte_t* oligo = (oligodendrocyte_t*) nimcp_malloc(sizeof(oligodendrocyte_t));
    if (!oligo) {
        return NULL;
    }

    memset(oligo, 0, sizeof(oligodendrocyte_t));

    oligo->id = id;
    oligo->max_neurons = max_axons;
    oligo->num_myelinated_neurons = 0;

    // Allocate arrays
    oligo->myelinated_neuron_ids = (uint32_t*) nimcp_malloc(max_axons * sizeof(uint32_t));
    oligo->myelination_levels = (float*) nimcp_malloc(max_axons * sizeof(float));
    oligo->neuron_activity_history = (float*) nimcp_malloc(max_axons * sizeof(float));
    oligo->last_spike_times = (uint64_t*) nimcp_malloc(max_axons * sizeof(uint64_t));

    if (!oligo->myelinated_neuron_ids || !oligo->myelination_levels ||
        !oligo->neuron_activity_history || !oligo->last_spike_times) {
        oligodendrocyte_destroy(oligo);
        return NULL;
    }

    // Initialize arrays
    for (uint32_t i = 0; i < max_axons; i++) {
        oligo->myelinated_neuron_ids[i] = UINT32_MAX;
        oligo->myelination_levels[i] = 0.0f;
        oligo->neuron_activity_history[i] = 0.0f;
        oligo->last_spike_times[i] = 0;
    }

    // Initial metabolic state
    oligo->atp_level = 1.0f; // Fully energized
    oligo->metabolic_cost = 0.0f;
    oligo->max_myelination_capacity = (float)max_axons; // Can fully myelinate all axons

    // Remodeling parameters
    oligo->last_remodeling_time = nimcp_time_monotonic_us();
    oligo->remodeling_interval_ms = NIMCP_OLIGO_REMODEL_TAU_S * 1000.0f;

    // Spatial position (default to origin)
    oligo->position[0] = 0.0f;
    oligo->position[1] = 0.0f;
    oligo->position[2] = 0.0f;

    // Initialize lock
    nimcp_spinlock_init(&oligo->lock);

    return oligo;
}

void oligodendrocyte_destroy(oligodendrocyte_t* oligo)
{
    if (!oligo) return;

    if (oligo->myelinated_neuron_ids) {
        nimcp_free(oligo->myelinated_neuron_ids);
    }
    if (oligo->myelination_levels) {
        nimcp_free(oligo->myelination_levels);
    }
    if (oligo->neuron_activity_history) {
        nimcp_free(oligo->neuron_activity_history);
    }
    if (oligo->last_spike_times) {
        nimcp_free(oligo->last_spike_times);
    }

    nimcp_free(oligo);
}

// ============================================================================
// NEURON ASSIGNMENT & MYELINATION
// ============================================================================

nimcp_result_t oligodendrocyte_assign_neuron(oligodendrocyte_t* oligo, uint32_t neuron_id)
{
    if (!oligo) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_spinlock_lock(&oligo->lock);

    // Check for duplicate
    for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
        if (oligo->myelinated_neuron_ids[i] == neuron_id) {
            nimcp_spinlock_unlock(&oligo->lock);
            return NIMCP_SUCCESS; // Already assigned
        }
    }

    // Check capacity
    if (oligo->num_myelinated_neurons >= oligo->max_neurons) {
        nimcp_spinlock_unlock(&oligo->lock);
        return NIMCP_ERROR_INVALID_PARAM; // At capacity
    }

    // Assign neuron
    uint32_t idx = oligo->num_myelinated_neurons;
    oligo->myelinated_neuron_ids[idx] = neuron_id;
    oligo->myelination_levels[idx] = 0.0f; // Start UNMYELINATED - activity will increase this
    oligo->neuron_activity_history[idx] = 0.0f;
    oligo->last_spike_times[idx] = 0;

    oligo->num_myelinated_neurons++;

    nimcp_spinlock_unlock(&oligo->lock);
    return NIMCP_SUCCESS;
}

float oligodendrocyte_get_myelination_level(oligodendrocyte_t* oligo, uint32_t neuron_id)
{
    if (!oligo) return 0.0f;

    nimcp_spinlock_lock(&oligo->lock);

    for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
        if (oligo->myelinated_neuron_ids[i] == neuron_id) {
            float level = oligo->myelination_levels[i];
            nimcp_spinlock_unlock(&oligo->lock);
            return level;
        }
    }

    nimcp_spinlock_unlock(&oligo->lock);
    return 0.0f; // Not found
}

float oligodendrocyte_compute_conduction_velocity(oligodendrocyte_t* oligo,
                                                   uint32_t neuron_id,
                                                   float base_velocity)
{
    if (!oligo) return base_velocity;

    float myelin_level = oligodendrocyte_get_myelination_level(oligo, neuron_id);

    // velocity = base × (1 + myelin × (multiplier - 1))
    float multiplier = NIMCP_OLIGO_MYELIN_MULTIPLIER;
    float velocity = base_velocity * (1.0f + myelin_level * (multiplier - 1.0f));

    return velocity;
}

// ============================================================================
// ACTIVITY TRACKING & ADAPTIVE MYELINATION
// ============================================================================

void oligodendrocyte_track_activity(oligodendrocyte_t* oligo,
                                    uint32_t neuron_id,
                                    float activity,
                                    uint64_t timestamp)
{
    if (!oligo) return;

    // Clamp activity to reasonable range
    if (activity < 0.0f) activity = 0.0f;
    if (activity > 100.0f) activity = 100.0f;

    nimcp_spinlock_lock(&oligo->lock);

    for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
        if (oligo->myelinated_neuron_ids[i] == neuron_id) {
            // Update rolling average (exponential moving average)
            // Higher alpha = faster adaptation to new activity patterns
            float alpha = 0.3f; // Smoothing factor (0.3 = adapt within ~5-10 samples)
            oligo->neuron_activity_history[i] =
                alpha * activity + (1.0f - alpha) * oligo->neuron_activity_history[i];

            oligo->last_spike_times[i] = timestamp;
            break;
        }
    }

    nimcp_spinlock_unlock(&oligo->lock);
}

void oligodendrocyte_remodel_myelination(oligodendrocyte_t* oligo, float dt)
{
    if (!oligo || dt <= 0.0f) return;

    nimcp_spinlock_lock(&oligo->lock);

    // Adaptive myelination based on activity
    for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
        float activity = oligo->neuron_activity_history[i];
        float current_myelin = oligo->myelination_levels[i];

        // Target myelination based on activity
        float activity_normalized = fminf(1.0f, activity / 10.0f); // Normalize to 0-1
        float target_myelin = activity_normalized;

        // Gradual change towards target (rate limited by time constant and ATP)
        float tau = NIMCP_OLIGO_REMODEL_TAU_S;
        float remodel_rate = dt / tau;

        // ATP limits remodeling rate
        float atp_factor = fmaxf(0.1f, oligo->atp_level); // Minimum 10% rate even at low ATP
        remodel_rate *= atp_factor;

        float delta = (target_myelin - current_myelin) * remodel_rate;
        oligo->myelination_levels[i] += delta;

        // Clamp to valid range
        oligo->myelination_levels[i] = fmaxf(0.0f, fminf(1.0f, oligo->myelination_levels[i]));
    }

    // Enforce capacity constraint
    float total_myelin = oligodendrocyte_get_total_myelination(oligo);
    if (total_myelin > oligo->max_myelination_capacity) {
        // Scale down all myelination proportionally
        float scale = oligo->max_myelination_capacity / total_myelin;
        for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
            oligo->myelination_levels[i] *= scale;
        }
    }

    oligo->last_remodeling_time = nimcp_time_monotonic_us();

    nimcp_spinlock_unlock(&oligo->lock);
}

// ============================================================================
// METABOLIC MANAGEMENT
// ============================================================================

void oligodendrocyte_update_atp(oligodendrocyte_t* oligo, float dt)
{
    if (!oligo || dt <= 0.0f) return;

    nimcp_spinlock_lock(&oligo->lock);

    // ATP cost proportional to total myelination
    float total_myelin = oligodendrocyte_get_total_myelination(oligo);
    float cost = total_myelin * NIMCP_OLIGO_ATP_COST_PER_MYELIN;

    oligo->metabolic_cost = cost;

    // Update ATP: regeneration - cost
    oligo->atp_level += dt * (NIMCP_OLIGO_ATP_REGEN_RATE - cost);

    // Clamp to 0-1
    oligo->atp_level = fmaxf(0.0f, fminf(1.0f, oligo->atp_level));

    nimcp_spinlock_unlock(&oligo->lock);
}

// ============================================================================
// NETWORK MANAGEMENT
// ============================================================================

oligodendrocyte_network_t* oligodendrocyte_network_create(uint32_t capacity)
{
    if (capacity == 0) return NULL;

    oligodendrocyte_network_t* network =
        (oligodendrocyte_network_t*) nimcp_malloc(sizeof(oligodendrocyte_network_t));
    if (!network) return NULL;

    memset(network, 0, sizeof(oligodendrocyte_network_t));

    network->capacity = capacity;
    network->num_oligodendrocytes = 0;

    network->oligodendrocytes =
        (oligodendrocyte_t**) nimcp_malloc(capacity * sizeof(oligodendrocyte_t*));
    if (!network->oligodendrocytes) {
        nimcp_free(network);
        return NULL;
    }

    for (uint32_t i = 0; i < capacity; i++) {
        network->oligodendrocytes[i] = NULL;
    }

    // Global parameters
    network->base_conduction_velocity = NIMCP_OLIGO_BASE_VELOCITY_MS;
    network->myelinated_velocity_multiplier = NIMCP_OLIGO_MYELIN_MULTIPLIER;
    network->activity_threshold = NIMCP_OLIGO_ACTIVITY_THRESHOLD_HZ;

    nimcp_mutex_init(&network->lock, NULL);

    return network;
}

void oligodendrocyte_network_destroy(oligodendrocyte_network_t* network)
{
    if (!network) return;

    // Destroy all oligodendrocytes
    for (uint32_t i = 0; i < network->num_oligodendrocytes; i++) {
        if (network->oligodendrocytes[i]) {
            oligodendrocyte_destroy(network->oligodendrocytes[i]);
        }
    }

    if (network->oligodendrocytes) {
        nimcp_free(network->oligodendrocytes);
    }

    nimcp_mutex_destroy(&network->lock);
    nimcp_free(network);
}

nimcp_result_t oligodendrocyte_network_add(oligodendrocyte_network_t* network,
                                            oligodendrocyte_t* oligo)
{
    if (!network || !oligo) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&network->lock);

    if (network->num_oligodendrocytes >= network->capacity) {
        nimcp_mutex_unlock(&network->lock);
        return NIMCP_ERROR_INVALID_PARAM; // At capacity
    }

    network->oligodendrocytes[network->num_oligodendrocytes] = oligo;
    network->num_oligodendrocytes++;

    nimcp_mutex_unlock(&network->lock);
    return NIMCP_SUCCESS;
}

void oligodendrocyte_network_step(oligodendrocyte_network_t* network, float dt)
{
    if (!network || dt <= 0.0f) return;

    nimcp_mutex_lock(&network->lock);

    for (uint32_t i = 0; i < network->num_oligodendrocytes; i++) {
        oligodendrocyte_t* oligo = network->oligodendrocytes[i];
        if (oligo) {
            // Remodel myelination based on activity
            oligodendrocyte_remodel_myelination(oligo, dt);

            // Update ATP
            oligodendrocyte_update_atp(oligo, dt);
        }
    }

    nimcp_mutex_unlock(&network->lock);
}

oligodendrocyte_t* oligodendrocyte_network_find_by_neuron(oligodendrocyte_network_t* network,
                                                           uint32_t neuron_id)
{
    if (!network) return NULL;

    nimcp_mutex_lock(&network->lock);

    for (uint32_t i = 0; i < network->num_oligodendrocytes; i++) {
        oligodendrocyte_t* oligo = network->oligodendrocytes[i];
        if (!oligo) continue;

        for (uint32_t j = 0; j < oligo->num_myelinated_neurons; j++) {
            if (oligo->myelinated_neuron_ids[j] == neuron_id) {
                nimcp_mutex_unlock(&network->lock);
                return oligo;
            }
        }
    }

    nimcp_mutex_unlock(&network->lock);
    return NULL;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

float oligodendrocyte_get_total_myelination(oligodendrocyte_t* oligo)
{
    if (!oligo) return 0.0f;

    float total = 0.0f;
    for (uint32_t i = 0; i < oligo->num_myelinated_neurons; i++) {
        total += oligo->myelination_levels[i];
    }

    return total;
}
