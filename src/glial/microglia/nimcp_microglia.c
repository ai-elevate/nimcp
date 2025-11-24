/**
 * @file nimcp_microglia.c
 * @brief Enhanced Microglia Implementation with Mathematical Algorithms
 *
 * FEATURES:
 * - KD-tree spatial indexing for O(log n) queries
 * - RK4 ODE integration for state dynamics
 * - Centrality-protected pruning
 * - Complement cascade (C1q/C3) tagging
 * - Cytokine signaling system
 * - Signal filtering for stable activity assessment
 *
 * @author NIMCP Development Team
 * @date 2025-11-24
 * @version 2.0.0
 */

#include "nimcp_microglia.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/numerical/nimcp_integration.h"
#include <string.h>
#include <math.h>
#include <float.h>

//=============================================================================
// INTERNAL CONSTANTS
//=============================================================================

/** @brief C1q to C3 conversion time (seconds) */
#define C1Q_TO_C3_CONVERSION_TIME_S 2.0f

/** @brief Activity filter alpha (derived from cutoff) */
#define DEFAULT_FILTER_ALPHA 0.1f

/** @brief State ODE indices */
#define STATE_IDX_INFLAMMATION 0
#define STATE_IDX_ACTIVATION 1
#define STATE_IDX_PROCESS 2
#define STATE_IDX_ENERGY 3

//=============================================================================
// INTERNAL HELPER FUNCTIONS
//=============================================================================

/**
 * @brief Find synapse index in monitoring array
 */
static int32_t find_synapse_index(const microglia_t* mg, uint32_t synapse_id)
{
    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->synapses[i].synapse_id == synapse_id) {
            return (int32_t)i;
        }
    }
    return -1;
}

/**
 * @brief Clamp value to range
 */
static inline float clamp_f(float value, float min_val, float max_val)
{
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * @brief Compute distance between two 3D points
 */
static inline float distance_3d(const float* p1, const float* p2)
{
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/**
 * @brief State dynamics derivative function for RK4
 *
 * State vector: [inflammation, activation, process, energy]
 */
static void state_derivatives(const float* state, float t, void* params, float* deriv)
{
    (void)t;  // Unused
    microglia_t* mg = (microglia_t*)params;

    float inflammation = state[STATE_IDX_INFLAMMATION];
    float activation = state[STATE_IDX_ACTIVATION];
    float process = state[STATE_IDX_PROCESS];
    float energy = state[STATE_IDX_ENERGY];

    // Inflammation dynamics: driven by external input, decays naturally
    float inflammation_input = mg->inflammation_level;
    float inflammation_decay = 0.2f;
    deriv[STATE_IDX_INFLAMMATION] = inflammation_input - inflammation_decay * inflammation;

    // Activation dynamics: follows inflammation with delay
    float activation_rate = 0.5f;
    float activation_target = inflammation;  // Activation tracks inflammation
    deriv[STATE_IDX_ACTIVATION] = activation_rate * (activation_target - activation);

    // Process extension dynamics: depends on state
    float process_rate = NIMCP_MICROGLIA_PROCESS_EXTENSION_RATE / 100.0f;
    if (inflammation < NIMCP_MICROGLIA_ACTIVATION_THRESHOLD) {
        // Ramified: extend processes
        deriv[STATE_IDX_PROCESS] = process_rate * (1.0f - process);
    } else {
        // Activated/Phagocytic: retract processes
        deriv[STATE_IDX_PROCESS] = -process_rate * 2.0f * process;
    }

    // Energy dynamics: depletes with activity, regenerates at rest
    float energy_cost = 0.1f * activation;
    float energy_regen = 0.05f * (1.0f - activation);
    deriv[STATE_IDX_ENERGY] = energy_regen - energy_cost;
}

/**
 * @brief Determine state from inflammation level
 */
static microglia_state_t compute_state_from_inflammation(float inflammation)
{
    if (inflammation >= NIMCP_MICROGLIA_PHAGOCYTIC_THRESHOLD) {
        return MICROGLIA_STATE_PHAGOCYTIC;
    } else if (inflammation >= NIMCP_MICROGLIA_ACTIVATION_THRESHOLD) {
        return MICROGLIA_STATE_ACTIVATED;
    } else {
        return MICROGLIA_STATE_RAMIFIED;
    }
}

/**
 * @brief Compute effective pruning threshold with centrality protection
 */
static float compute_effective_threshold(const microglia_t* mg,
                                          const monitored_synapse_t* syn)
{
    float base_threshold = mg->pruning_threshold;

    // Centrality protection: higher centrality = higher threshold needed to prune
    if (syn->protected_by_centrality &&
        syn->centrality_score >= NIMCP_CENTRALITY_PROTECTION_MIN) {
        base_threshold *= (1.0f + syn->centrality_score * NIMCP_CENTRALITY_PROTECTION_FACTOR);
    }

    // C3-tagged synapses are easier to prune (lower effective threshold)
    if (syn->complement.tag == COMPLEMENT_C3) {
        base_threshold *= 1.5f;  // Raise threshold meaning lower activity is OK to prune
    }

    return base_threshold;
}

//=============================================================================
// CREATION & DESTRUCTION
//=============================================================================

microglia_t* microglia_create(uint32_t id, float x, float y, float z,
                               float surveillance_radius)
{
    if (surveillance_radius <= 0.0f) {
        return NULL;
    }

    microglia_t* mg = (microglia_t*)nimcp_malloc(sizeof(microglia_t));
    if (!mg) {
        return NULL;
    }

    memset(mg, 0, sizeof(microglia_t));

    // Basic properties
    mg->id = id;
    mg->position[0] = x;
    mg->position[1] = y;
    mg->position[2] = z;
    mg->surveillance_radius = surveillance_radius;
    mg->process_extension = 1.0f;  // Start fully extended (ramified)

    // State dynamics
    mg->state = MICROGLIA_STATE_RAMIFIED;
    mg->inflammation_level = 0.0f;
    mg->state_variables[STATE_IDX_INFLAMMATION] = 0.0f;
    mg->state_variables[STATE_IDX_ACTIVATION] = 0.0f;
    mg->state_variables[STATE_IDX_PROCESS] = 1.0f;
    mg->state_variables[STATE_IDX_ENERGY] = 1.0f;

    // Cytokine state
    memset(&mg->cytokines, 0, sizeof(microglia_cytokine_state_t));
    mg->cytokines.last_update_time = nimcp_time_monotonic_us();

    // Allocate enhanced synapse array
    mg->max_monitored_synapses = NIMCP_MICROGLIA_DEFAULT_CAPACITY;
    mg->num_monitored_synapses = 0;

    mg->synapses = (monitored_synapse_t*)nimcp_malloc(
        mg->max_monitored_synapses * sizeof(monitored_synapse_t));
    if (!mg->synapses) {
        nimcp_free(mg);
        return NULL;
    }
    memset(mg->synapses, 0, mg->max_monitored_synapses * sizeof(monitored_synapse_t));

    // Legacy arrays for backward compatibility
    mg->monitored_synapse_ids = (uint32_t*)nimcp_malloc(
        mg->max_monitored_synapses * sizeof(uint32_t));
    mg->synapse_activity_scores = (float*)nimcp_malloc(
        mg->max_monitored_synapses * sizeof(float));
    mg->last_activity_times = (uint64_t*)nimcp_malloc(
        mg->max_monitored_synapses * sizeof(uint64_t));

    if (!mg->monitored_synapse_ids || !mg->synapse_activity_scores ||
        !mg->last_activity_times) {
        microglia_destroy(mg);
        return NULL;
    }

    for (uint32_t i = 0; i < mg->max_monitored_synapses; i++) {
        mg->monitored_synapse_ids[i] = UINT32_MAX;
        mg->synapse_activity_scores[i] = 0.0f;
        mg->last_activity_times[i] = 0;
    }

    // Pruning parameters
    mg->pruning_threshold = NIMCP_MICROGLIA_PRUNING_THRESHOLD;
    mg->pruning_rate = NIMCP_MICROGLIA_PRUNING_RATE;
    mg->last_pruning_time = nimcp_time_monotonic_us();
    mg->total_synapses_pruned = 0;
    mg->total_c1q_tags = 0;
    mg->total_c3_conversions = 0;
    mg->protected_from_pruning = 0;

    // Initialize lock
    nimcp_spinlock_init(&mg->lock);

    return mg;
}

void microglia_destroy(microglia_t* mg)
{
    if (!mg) return;

    if (mg->synapses) {
        nimcp_free(mg->synapses);
    }
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

microglia_network_config_t microglia_network_default_config(void)
{
    microglia_network_config_t config = {
        .capacity = 100,
        .pruning_threshold = NIMCP_MICROGLIA_PRUNING_THRESHOLD,
        .pruning_rate = NIMCP_MICROGLIA_PRUNING_RATE,
        .surveillance_radius = NIMCP_MICROGLIA_SURVEILLANCE_RADIUS_UM,
        .enable_centrality_protection = true,
        .enable_complement_cascade = true,
        .enable_cytokine_signaling = true,
        .enable_state_dynamics = true,
        .filter_cutoff_hz = 1.0f
    };
    return config;
}

microglia_network_t* microglia_network_create_enhanced(
    const microglia_network_config_t* config)
{
    if (!config || config->capacity == 0) {
        return NULL;
    }

    microglia_network_t* network = (microglia_network_t*)nimcp_malloc(
        sizeof(microglia_network_t));
    if (!network) {
        return NULL;
    }

    memset(network, 0, sizeof(microglia_network_t));

    network->capacity = config->capacity;
    network->num_microglia = 0;

    network->microglia = (microglia_t**)nimcp_malloc(
        config->capacity * sizeof(microglia_t*));
    if (!network->microglia) {
        nimcp_free(network);
        return NULL;
    }

    for (uint32_t i = 0; i < config->capacity; i++) {
        network->microglia[i] = NULL;
    }

    // Create KD-trees for spatial indexing
    network->microglia_tree = kdtree_create();
    network->synapse_tree = kdtree_create();
    network->spatial_index_valid = false;

    // Global parameters
    network->global_pruning_threshold = config->pruning_threshold;
    network->min_activity_window_ms = NIMCP_MICROGLIA_MIN_ACTIVITY_WINDOW_MS;
    network->global_inflammation = 0.0f;

    // Activity filter
    network->filter_cutoff_hz = config->filter_cutoff_hz;
    // alpha = 2πf_c / (2πf_c + f_s), simplified for ~1000Hz sample rate
    network->filter_alpha = 2.0f * M_PI * config->filter_cutoff_hz /
                            (2.0f * M_PI * config->filter_cutoff_hz + 1000.0f);
    if (network->filter_alpha < 0.01f) network->filter_alpha = 0.01f;
    if (network->filter_alpha > 1.0f) network->filter_alpha = 1.0f;

    // Centrality (will be populated later)
    network->synapse_centrality = NULL;
    network->num_centrality_scores = 0;
    network->centrality_valid = false;

    // Cytokine field (placeholder - could be 3D grid)
    network->global_cytokine_field = NULL;
    network->cytokine_field_size = 0;

    nimcp_mutex_init(&network->lock, NULL);

    return network;
}

microglia_network_t* microglia_network_create(uint32_t capacity)
{
    microglia_network_config_t config = microglia_network_default_config();
    config.capacity = capacity;
    return microglia_network_create_enhanced(&config);
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

    // Destroy KD-trees
    if (network->microglia_tree) {
        kdtree_destroy(network->microglia_tree);
    }
    if (network->synapse_tree) {
        kdtree_destroy(network->synapse_tree);
    }

    // Free centrality scores
    if (network->synapse_centrality) {
        nimcp_free(network->synapse_centrality);
    }

    // Free cytokine field
    if (network->global_cytokine_field) {
        nimcp_free(network->global_cytokine_field);
    }

    nimcp_mutex_destroy(&network->lock);
    nimcp_free(network);
}

//=============================================================================
// SYNAPSE MONITORING
//=============================================================================

nimcp_result_t microglia_monitor_synapse_at(microglia_t* mg, uint32_t synapse_id,
                                             float x, float y, float z)
{
    if (!mg) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_spinlock_lock(&mg->lock);

    // Check for duplicate
    int32_t existing_idx = find_synapse_index(mg, synapse_id);
    if (existing_idx >= 0) {
        // Update position
        mg->synapses[existing_idx].position[0] = x;
        mg->synapses[existing_idx].position[1] = y;
        mg->synapses[existing_idx].position[2] = z;
        nimcp_spinlock_unlock(&mg->lock);
        return NIMCP_SUCCESS;
    }

    // Check capacity
    if (mg->num_monitored_synapses >= mg->max_monitored_synapses) {
        nimcp_spinlock_unlock(&mg->lock);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Add synapse
    uint32_t idx = mg->num_monitored_synapses;
    monitored_synapse_t* syn = &mg->synapses[idx];

    syn->synapse_id = synapse_id;
    syn->position[0] = x;
    syn->position[1] = y;
    syn->position[2] = z;
    syn->activity_score = 0.0f;
    syn->filtered_activity = 0.0f;
    syn->last_activity_time = nimcp_time_monotonic_us();
    syn->complement.tag = COMPLEMENT_NONE;
    syn->complement.tag_strength = 0.0f;
    syn->complement.tag_time = 0;
    syn->centrality_score = 0.0f;
    syn->protected_by_centrality = false;

    // Update legacy arrays
    mg->monitored_synapse_ids[idx] = synapse_id;
    mg->synapse_activity_scores[idx] = 0.0f;
    mg->last_activity_times[idx] = syn->last_activity_time;

    mg->num_monitored_synapses++;

    nimcp_spinlock_unlock(&mg->lock);
    return NIMCP_SUCCESS;
}

nimcp_result_t microglia_monitor_synapse(microglia_t* mg, uint32_t synapse_id)
{
    // Default position at microglia location
    if (!mg) return NIMCP_ERROR_INVALID_PARAM;
    return microglia_monitor_synapse_at(mg, synapse_id,
                                         mg->position[0], mg->position[1], mg->position[2]);
}

void microglia_track_synapse_activity(microglia_t* mg, uint32_t synapse_id,
                                       float activity, uint64_t timestamp)
{
    if (!mg) return;

    activity = clamp_f(activity, 0.0f, 10.0f);

    nimcp_spinlock_lock(&mg->lock);

    int32_t idx = find_synapse_index(mg, synapse_id);
    if (idx >= 0) {
        monitored_synapse_t* syn = &mg->synapses[idx];

        // Apply low-pass filter to activity
        float alpha = DEFAULT_FILTER_ALPHA;
        syn->filtered_activity = alpha * activity + (1.0f - alpha) * syn->filtered_activity;

        // Update raw activity score with EMA
        if (syn->activity_score < 0.001f) {
            syn->activity_score = activity;
        } else {
            float ema_alpha = 0.3f;
            syn->activity_score = ema_alpha * activity + (1.0f - ema_alpha) * syn->activity_score;
        }

        syn->last_activity_time = timestamp;

        // Update legacy arrays
        mg->synapse_activity_scores[idx] = syn->activity_score;
        mg->last_activity_times[idx] = timestamp;
    }

    nimcp_spinlock_unlock(&mg->lock);
}

void microglia_update_activity_scores(microglia_t* mg, uint64_t current_time)
{
    if (!mg) return;

    nimcp_spinlock_lock(&mg->lock);

    float tau_us = NIMCP_MICROGLIA_ACTIVITY_DECAY_TAU_S * 1000000.0f;

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        monitored_synapse_t* syn = &mg->synapses[i];
        uint64_t dt_us = current_time - syn->last_activity_time;
        float dt_s = (float)dt_us / 1000000.0f;

        // Exponential decay
        float decay_factor = expf(-dt_s * 1000000.0f / tau_us);
        syn->activity_score *= decay_factor;
        syn->filtered_activity *= decay_factor;

        // Clamp to zero
        if (syn->activity_score < 0.001f) {
            syn->activity_score = 0.0f;
        }
        if (syn->filtered_activity < 0.001f) {
            syn->filtered_activity = 0.0f;
        }

        // Update legacy arrays
        mg->synapse_activity_scores[i] = syn->activity_score;
    }

    nimcp_spinlock_unlock(&mg->lock);
}

float microglia_get_synapse_activity_score(microglia_t* mg, uint32_t synapse_id)
{
    if (!mg) return 0.0f;

    nimcp_spinlock_lock(&mg->lock);

    int32_t idx = find_synapse_index(mg, synapse_id);
    float score = (idx >= 0) ? mg->synapses[idx].activity_score : 0.0f;

    nimcp_spinlock_unlock(&mg->lock);
    return score;
}

void microglia_set_synapse_centrality(microglia_t* mg, uint32_t synapse_id,
                                       float centrality)
{
    if (!mg) return;

    centrality = clamp_f(centrality, 0.0f, 1.0f);

    nimcp_spinlock_lock(&mg->lock);

    int32_t idx = find_synapse_index(mg, synapse_id);
    if (idx >= 0) {
        mg->synapses[idx].centrality_score = centrality;
        mg->synapses[idx].protected_by_centrality =
            (centrality >= NIMCP_CENTRALITY_PROTECTION_MIN);
    }

    nimcp_spinlock_unlock(&mg->lock);
}

//=============================================================================
// STATE DYNAMICS (RK4 ODE)
//=============================================================================

void microglia_update_state_dynamics(microglia_t* mg, float dt)
{
    if (!mg || dt <= 0.0f) return;

    nimcp_spinlock_lock(&mg->lock);

    // Use RK4 integration for state dynamics
    float k1[4], k2[4], k3[4], k4[4];
    float temp_state[4];

    // k1 = f(state, t)
    state_derivatives(mg->state_variables, 0.0f, mg, k1);

    // k2 = f(state + dt/2 * k1, t + dt/2)
    for (int i = 0; i < 4; i++) {
        temp_state[i] = mg->state_variables[i] + 0.5f * dt * k1[i];
    }
    state_derivatives(temp_state, dt * 0.5f, mg, k2);

    // k3 = f(state + dt/2 * k2, t + dt/2)
    for (int i = 0; i < 4; i++) {
        temp_state[i] = mg->state_variables[i] + 0.5f * dt * k2[i];
    }
    state_derivatives(temp_state, dt * 0.5f, mg, k3);

    // k4 = f(state + dt * k3, t + dt)
    for (int i = 0; i < 4; i++) {
        temp_state[i] = mg->state_variables[i] + dt * k3[i];
    }
    state_derivatives(temp_state, dt, mg, k4);

    // Update state: state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (int i = 0; i < 4; i++) {
        mg->state_variables[i] += dt / 6.0f * (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }

    // Clamp values
    mg->state_variables[STATE_IDX_INFLAMMATION] =
        clamp_f(mg->state_variables[STATE_IDX_INFLAMMATION], 0.0f, 1.0f);
    mg->state_variables[STATE_IDX_ACTIVATION] =
        clamp_f(mg->state_variables[STATE_IDX_ACTIVATION], 0.0f, 1.0f);
    mg->state_variables[STATE_IDX_PROCESS] =
        clamp_f(mg->state_variables[STATE_IDX_PROCESS], 0.0f, 1.0f);
    mg->state_variables[STATE_IDX_ENERGY] =
        clamp_f(mg->state_variables[STATE_IDX_ENERGY], 0.0f, 1.0f);

    // Update derived values
    mg->process_extension = mg->state_variables[STATE_IDX_PROCESS];
    mg->state = compute_state_from_inflammation(mg->state_variables[STATE_IDX_INFLAMMATION]);

    nimcp_spinlock_unlock(&mg->lock);
}

microglia_state_t microglia_get_state(const microglia_t* mg)
{
    if (!mg) return MICROGLIA_STATE_RAMIFIED;
    return mg->state;
}

void microglia_set_inflammation(microglia_t* mg, float inflammation)
{
    if (!mg) return;

    nimcp_spinlock_lock(&mg->lock);
    mg->inflammation_level = clamp_f(inflammation, 0.0f, 1.0f);
    nimcp_spinlock_unlock(&mg->lock);
}

float microglia_get_process_extension(const microglia_t* mg)
{
    if (!mg) return 0.0f;
    return mg->process_extension;
}

//=============================================================================
// COMPLEMENT CASCADE
//=============================================================================

uint32_t microglia_apply_complement_tags(microglia_t* mg, uint64_t current_time)
{
    if (!mg) return 0;

    nimcp_spinlock_lock(&mg->lock);

    uint32_t newly_tagged = 0;
    float c1q_threshold = NIMCP_COMPLEMENT_C1Q_THRESHOLD;
    uint64_t conversion_time_us = (uint64_t)(C1Q_TO_C3_CONVERSION_TIME_S * 1000000.0f);

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        monitored_synapse_t* syn = &mg->synapses[i];

        // Only tag low-activity synapses
        if (syn->filtered_activity < c1q_threshold) {
            if (syn->complement.tag == COMPLEMENT_NONE) {
                // Apply C1q tag
                syn->complement.tag = COMPLEMENT_C1Q;
                syn->complement.tag_strength = 1.0f - (syn->filtered_activity / c1q_threshold);
                syn->complement.tag_time = current_time;
                mg->total_c1q_tags++;
                newly_tagged++;
            } else if (syn->complement.tag == COMPLEMENT_C1Q) {
                // Check for C3 conversion
                uint64_t tag_age = current_time - syn->complement.tag_time;
                if (tag_age >= conversion_time_us) {
                    syn->complement.tag = COMPLEMENT_C3;
                    syn->complement.tag_strength = 1.0f;
                    mg->total_c3_conversions++;
                }
            }
        } else {
            // Activity recovered - decay tag
            if (syn->complement.tag != COMPLEMENT_NONE) {
                syn->complement.tag_strength -= 0.1f;
                if (syn->complement.tag_strength <= 0.0f) {
                    syn->complement.tag = COMPLEMENT_NONE;
                    syn->complement.tag_strength = 0.0f;
                }
            }
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
    return newly_tagged;
}

complement_tag_t microglia_get_complement_tag(const microglia_t* mg, uint32_t synapse_id)
{
    if (!mg) return COMPLEMENT_NONE;

    // Note: Not locking for const read - may need adjustment for strict thread safety
    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->synapses[i].synapse_id == synapse_id) {
            return mg->synapses[i].complement.tag;
        }
    }
    return COMPLEMENT_NONE;
}

void microglia_decay_complement_tags(microglia_t* mg, float dt)
{
    if (!mg || dt <= 0.0f) return;

    nimcp_spinlock_lock(&mg->lock);

    float decay = NIMCP_COMPLEMENT_DECAY_RATE * dt;

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        monitored_synapse_t* syn = &mg->synapses[i];
        if (syn->complement.tag != COMPLEMENT_NONE) {
            syn->complement.tag_strength -= decay;
            if (syn->complement.tag_strength <= 0.0f) {
                syn->complement.tag = COMPLEMENT_NONE;
                syn->complement.tag_strength = 0.0f;
            }
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
}

//=============================================================================
// CYTOKINE SIGNALING
//=============================================================================

void microglia_update_cytokines(microglia_t* mg, float dt)
{
    if (!mg || dt <= 0.0f) return;

    nimcp_spinlock_lock(&mg->lock);

    // Production rates based on state
    float production[NIMCP_CYTOKINE_COUNT] = {0};

    switch (mg->state) {
        case MICROGLIA_STATE_RAMIFIED:
            // Low baseline production
            production[CYTOKINE_IL1B] = 0.01f;
            production[CYTOKINE_TNFA] = 0.01f;
            production[CYTOKINE_IL6] = 0.01f;
            production[CYTOKINE_IL10] = 0.02f;
            production[CYTOKINE_TGFB] = 0.02f;
            break;

        case MICROGLIA_STATE_ACTIVATED:
            // High pro-inflammatory
            production[CYTOKINE_IL1B] = 0.5f;
            production[CYTOKINE_TNFA] = 0.4f;
            production[CYTOKINE_IL6] = 0.3f;
            production[CYTOKINE_IL10] = 0.05f;
            production[CYTOKINE_TGFB] = 0.05f;
            break;

        case MICROGLIA_STATE_PHAGOCYTIC:
            // Resolution phase - anti-inflammatory
            production[CYTOKINE_IL1B] = 0.1f;
            production[CYTOKINE_TNFA] = 0.1f;
            production[CYTOKINE_IL6] = 0.1f;
            production[CYTOKINE_IL10] = 0.5f;
            production[CYTOKINE_TGFB] = 0.4f;
            break;
    }

    // Update concentrations: dC/dt = production - decay * C
    for (int i = 0; i < NIMCP_CYTOKINE_COUNT; i++) {
        mg->cytokines.production_rates[i] = production[i];
        float dC = production[i] - NIMCP_CYTOKINE_DECAY_RATE * mg->cytokines.concentrations[i];
        mg->cytokines.concentrations[i] += dC * dt;
        mg->cytokines.concentrations[i] = clamp_f(mg->cytokines.concentrations[i],
                                                   0.0f, NIMCP_CYTOKINE_MAX_CONCENTRATION);
    }

    mg->cytokines.last_update_time = nimcp_time_monotonic_us();

    nimcp_spinlock_unlock(&mg->lock);
}

float microglia_get_cytokine(const microglia_t* mg, cytokine_type_t type)
{
    if (!mg || type >= NIMCP_CYTOKINE_COUNT) return 0.0f;
    return mg->cytokines.concentrations[type];
}

void microglia_add_cytokine(microglia_t* mg, cytokine_type_t type, float amount)
{
    if (!mg || type >= NIMCP_CYTOKINE_COUNT) return;

    nimcp_spinlock_lock(&mg->lock);
    mg->cytokines.concentrations[type] = clamp_f(
        mg->cytokines.concentrations[type] + amount,
        0.0f, NIMCP_CYTOKINE_MAX_CONCENTRATION);
    nimcp_spinlock_unlock(&mg->lock);
}

float microglia_get_net_inflammation(const microglia_t* mg)
{
    if (!mg) return 0.0f;

    float pro = mg->cytokines.concentrations[CYTOKINE_IL1B] +
                mg->cytokines.concentrations[CYTOKINE_TNFA] +
                mg->cytokines.concentrations[CYTOKINE_IL6];

    float anti = mg->cytokines.concentrations[CYTOKINE_IL10] +
                 mg->cytokines.concentrations[CYTOKINE_TGFB];

    return pro - anti;
}

//=============================================================================
// PRUNING (Enhanced with Centrality Protection)
//=============================================================================

uint32_t microglia_identify_weak_synapses(microglia_t* mg,
                                           uint32_t* weak_synapse_ids,
                                           uint32_t max_count)
{
    if (!mg || !weak_synapse_ids || max_count == 0) return 0;

    nimcp_spinlock_lock(&mg->lock);

    uint32_t num_weak = 0;

    for (uint32_t i = 0; i < mg->num_monitored_synapses && num_weak < max_count; i++) {
        monitored_synapse_t* syn = &mg->synapses[i];
        float effective_threshold = compute_effective_threshold(mg, syn);

        // Use filtered activity for more stable decisions
        if (syn->filtered_activity < effective_threshold) {
            // Check centrality protection
            if (syn->protected_by_centrality &&
                syn->centrality_score >= NIMCP_CENTRALITY_PROTECTION_MIN) {
                // Only prune if C3 tagged (overrides protection)
                if (syn->complement.tag != COMPLEMENT_C3) {
                    mg->protected_from_pruning++;
                    continue;
                }
            }

            weak_synapse_ids[num_weak] = syn->synapse_id;
            num_weak++;
        }
    }

    nimcp_spinlock_unlock(&mg->lock);
    return num_weak;
}

uint32_t microglia_prune_weak_synapses(microglia_t* mg)
{
    if (!mg || mg->num_monitored_synapses == 0) return 0;

    nimcp_spinlock_lock(&mg->lock);

    uint32_t num_pruned = 0;
    uint32_t max_prune = (uint32_t)mg->pruning_rate;

    // Prioritize C3-tagged synapses
    uint32_t write_idx = 0;

    // First pass: prune C3-tagged weak synapses
    for (uint32_t read_idx = 0; read_idx < mg->num_monitored_synapses; read_idx++) {
        monitored_synapse_t* syn = &mg->synapses[read_idx];
        float effective_threshold = compute_effective_threshold(mg, syn);

        bool is_weak = syn->filtered_activity < effective_threshold;
        bool should_prune = is_weak && (num_pruned < max_prune);

        // Respect centrality protection unless C3-tagged
        if (should_prune && syn->protected_by_centrality &&
            syn->centrality_score >= NIMCP_CENTRALITY_PROTECTION_MIN &&
            syn->complement.tag != COMPLEMENT_C3) {
            should_prune = false;
            mg->protected_from_pruning++;
        }

        if (should_prune) {
            num_pruned++;
            // Don't copy - effectively removes
        } else {
            // Keep this synapse
            if (write_idx != read_idx) {
                mg->synapses[write_idx] = mg->synapses[read_idx];
                mg->monitored_synapse_ids[write_idx] = mg->monitored_synapse_ids[read_idx];
                mg->synapse_activity_scores[write_idx] = mg->synapse_activity_scores[read_idx];
                mg->last_activity_times[write_idx] = mg->last_activity_times[read_idx];
            }
            write_idx++;
        }
    }

    mg->num_monitored_synapses = write_idx;
    mg->total_synapses_pruned += num_pruned;
    mg->last_pruning_time = nimcp_time_monotonic_us();

    nimcp_spinlock_unlock(&mg->lock);
    return num_pruned;
}

bool microglia_should_prune_synapse(const microglia_t* mg, uint32_t synapse_id)
{
    if (!mg) return false;

    for (uint32_t i = 0; i < mg->num_monitored_synapses; i++) {
        if (mg->synapses[i].synapse_id == synapse_id) {
            const monitored_synapse_t* syn = &mg->synapses[i];
            float effective_threshold = compute_effective_threshold(mg, syn);

            if (syn->filtered_activity >= effective_threshold) {
                return false;  // Activity above threshold
            }

            // Check protection
            if (syn->protected_by_centrality &&
                syn->centrality_score >= NIMCP_CENTRALITY_PROTECTION_MIN &&
                syn->complement.tag != COMPLEMENT_C3) {
                return false;  // Protected by centrality
            }

            return true;  // Should prune
        }
    }

    return false;  // Not found
}

//=============================================================================
// NETWORK OPERATIONS
//=============================================================================

nimcp_result_t microglia_network_add(microglia_network_t* network, microglia_t* mg)
{
    if (!network || !mg) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&network->lock);

    if (network->num_microglia >= network->capacity) {
        nimcp_mutex_unlock(&network->lock);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    network->microglia[network->num_microglia] = mg;
    network->num_microglia++;
    network->spatial_index_valid = false;  // Need to rebuild KD-tree

    nimcp_mutex_unlock(&network->lock);
    return NIMCP_SUCCESS;
}

void microglia_network_rebuild_spatial_index(microglia_network_t* network)
{
    if (!network || network->num_microglia == 0) return;

    nimcp_mutex_lock(&network->lock);

    // Rebuild microglia KD-tree
    if (network->microglia_tree) {
        kdtree_destroy(network->microglia_tree);
    }
    network->microglia_tree = kdtree_create();

    if (network->microglia_tree && network->num_microglia > 0) {
        kdtree_point_t* points = (kdtree_point_t*)nimcp_malloc(
            network->num_microglia * sizeof(kdtree_point_t));
        void** user_data = (void**)nimcp_malloc(
            network->num_microglia * sizeof(void*));

        if (points && user_data) {
            for (uint32_t i = 0; i < network->num_microglia; i++) {
                microglia_t* mg = network->microglia[i];
                points[i][0] = mg->position[0];
                points[i][1] = mg->position[1];
                points[i][2] = mg->position[2];
                user_data[i] = mg;
            }

            kdtree_build(network->microglia_tree, points, user_data, network->num_microglia);
        }

        if (points) nimcp_free(points);
        if (user_data) nimcp_free(user_data);
    }

    network->spatial_index_valid = true;

    nimcp_mutex_unlock(&network->lock);
}

void microglia_network_update_centrality(microglia_network_t* network, void* synapse_graph)
{
    if (!network) return;

    nimcp_mutex_lock(&network->lock);

    // This would integrate with nimcp_centrality.h
    // For now, we'll set a placeholder that can be populated externally
    // In full integration, this would call:
    // NimcpCentralityScores* scores = nimcp_betweenness_centrality((NimcpGraph*)synapse_graph);

    // Mark centrality as needing external update
    network->centrality_valid = (synapse_graph != NULL);

    nimcp_mutex_unlock(&network->lock);
}

void microglia_network_step(microglia_network_t* network, uint64_t current_time)
{
    if (!network) return;

    nimcp_mutex_lock(&network->lock);

    // Compute dt from last step (default to 1ms if first step)
    static uint64_t last_step_time = 0;
    float dt_s;
    if (last_step_time == 0) {
        dt_s = 0.001f;
    } else {
        dt_s = (float)(current_time - last_step_time) / 1000000.0f;
        if (dt_s > 0.1f) dt_s = 0.1f;  // Cap at 100ms
        if (dt_s < 0.0001f) dt_s = 0.0001f;  // Min 0.1ms
    }
    last_step_time = current_time;

    // Process each microglia
    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg = network->microglia[i];
        if (!mg) continue;

        // 1. Update state dynamics (RK4)
        microglia_update_state_dynamics(mg, dt_s);

        // 2. Update cytokines
        microglia_update_cytokines(mg, dt_s);

        // 3. Apply complement tagging
        microglia_apply_complement_tags(mg, current_time);

        // 4. Decay complement tags
        microglia_decay_complement_tags(mg, dt_s);

        // 5. Update activity scores
        microglia_update_activity_scores(mg, current_time);

        // 6. Prune weak synapses
        microglia_prune_weak_synapses(mg);
    }

    // Diffuse cytokines between nearby microglia
    microglia_network_diffuse_cytokines(network, dt_s);

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

        if (find_synapse_index(mg, synapse_id) >= 0) {
            nimcp_mutex_unlock(&network->lock);
            return mg;
        }
    }

    nimcp_mutex_unlock(&network->lock);
    return NULL;
}

microglia_t* microglia_network_find_nearest(microglia_network_t* network,
                                             float x, float y, float z)
{
    if (!network) return NULL;

    nimcp_mutex_lock(&network->lock);

    // Use KD-tree if valid
    if (network->spatial_index_valid && network->microglia_tree) {
        kdtree_point_t query = {x, y, z};
        float dist_sq = 0.0f;

        void* nearest = kdtree_nearest(network->microglia_tree, query, &dist_sq);
        if (nearest) {
            nimcp_mutex_unlock(&network->lock);
            return (microglia_t*)nearest;
        }
    }

    // Fallback to linear search
    microglia_t* nearest = NULL;
    float min_dist = FLT_MAX;
    float query_pos[3] = {x, y, z};

    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg = network->microglia[i];
        if (!mg) continue;

        float dist = distance_3d(query_pos, mg->position);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = mg;
        }
    }

    nimcp_mutex_unlock(&network->lock);
    return nearest;
}

uint32_t microglia_network_find_in_radius(microglia_network_t* network,
                                           float x, float y, float z,
                                           float radius,
                                           microglia_t** results,
                                           uint32_t max_results)
{
    if (!network || !results || max_results == 0) return 0;

    nimcp_mutex_lock(&network->lock);

    uint32_t count = 0;
    float query_pos[3] = {x, y, z};
    float radius_sq = radius * radius;

    for (uint32_t i = 0; i < network->num_microglia && count < max_results; i++) {
        microglia_t* mg = network->microglia[i];
        if (!mg) continue;

        float dx = mg->position[0] - query_pos[0];
        float dy = mg->position[1] - query_pos[1];
        float dz = mg->position[2] - query_pos[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq <= radius_sq) {
            results[count++] = mg;
        }
    }

    nimcp_mutex_unlock(&network->lock);
    return count;
}

void microglia_network_diffuse_cytokines(microglia_network_t* network, float dt)
{
    if (!network || dt <= 0.0f || network->num_microglia < 2) return;

    // Simple diffusion: each microglia shares cytokines with nearby neighbors
    float diffusion_radius = 200.0f;  // µm
    float diffusion_rate = NIMCP_CYTOKINE_DIFFUSION_COEFF * dt / (diffusion_radius * diffusion_radius);

    // For each pair of nearby microglia, exchange cytokines
    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg1 = network->microglia[i];
        if (!mg1) continue;

        for (uint32_t j = i + 1; j < network->num_microglia; j++) {
            microglia_t* mg2 = network->microglia[j];
            if (!mg2) continue;

            float dist = distance_3d(mg1->position, mg2->position);
            if (dist > diffusion_radius) continue;

            // Distance-weighted diffusion
            float weight = diffusion_rate * (1.0f - dist / diffusion_radius);

            for (int c = 0; c < NIMCP_CYTOKINE_COUNT; c++) {
                float c1 = mg1->cytokines.concentrations[c];
                float c2 = mg2->cytokines.concentrations[c];
                float diff = (c2 - c1) * weight;

                nimcp_spinlock_lock(&mg1->lock);
                mg1->cytokines.concentrations[c] += diff;
                nimcp_spinlock_unlock(&mg1->lock);

                nimcp_spinlock_lock(&mg2->lock);
                mg2->cytokines.concentrations[c] -= diff;
                nimcp_spinlock_unlock(&mg2->lock);
            }
        }
    }
}

void microglia_network_get_stats(const microglia_network_t* network,
                                  microglia_network_stats_t* stats)
{
    if (!stats) return;

    memset(stats, 0, sizeof(microglia_network_stats_t));

    if (!network) return;

    stats->total_microglia = network->num_microglia;

    float total_inflammation = 0.0f;
    float total_activity = 0.0f;
    uint32_t total_synapses = 0;

    for (uint32_t i = 0; i < network->num_microglia; i++) {
        microglia_t* mg = network->microglia[i];
        if (!mg) continue;

        stats->total_monitored_synapses += mg->num_monitored_synapses;
        stats->total_pruned += mg->total_synapses_pruned;
        stats->total_c1q_tagged += mg->total_c1q_tags;
        stats->total_c3_tagged += mg->total_c3_conversions;
        stats->total_protected += mg->protected_from_pruning;

        switch (mg->state) {
            case MICROGLIA_STATE_RAMIFIED: stats->ramified_count++; break;
            case MICROGLIA_STATE_ACTIVATED: stats->activated_count++; break;
            case MICROGLIA_STATE_PHAGOCYTIC: stats->phagocytic_count++; break;
        }

        total_inflammation += mg->state_variables[STATE_IDX_INFLAMMATION];

        for (uint32_t j = 0; j < mg->num_monitored_synapses; j++) {
            total_activity += mg->synapses[j].activity_score;
            total_synapses++;
        }

        // Sum cytokines
        stats->total_pro_inflammatory +=
            mg->cytokines.concentrations[CYTOKINE_IL1B] +
            mg->cytokines.concentrations[CYTOKINE_TNFA] +
            mg->cytokines.concentrations[CYTOKINE_IL6];

        stats->total_anti_inflammatory +=
            mg->cytokines.concentrations[CYTOKINE_IL10] +
            mg->cytokines.concentrations[CYTOKINE_TGFB];
    }

    if (network->num_microglia > 0) {
        stats->avg_inflammation = total_inflammation / network->num_microglia;
    }

    if (total_synapses > 0) {
        stats->avg_activity_score = total_activity / total_synapses;
    }
}

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

uint32_t microglia_get_total_pruned(microglia_t* mg)
{
    if (!mg) return 0;
    return mg->total_synapses_pruned;
}

const char* microglia_state_to_string(microglia_state_t state)
{
    switch (state) {
        case MICROGLIA_STATE_RAMIFIED: return "Ramified";
        case MICROGLIA_STATE_ACTIVATED: return "Activated";
        case MICROGLIA_STATE_PHAGOCYTIC: return "Phagocytic";
        default: return "Unknown";
    }
}

const char* cytokine_type_to_string(cytokine_type_t type)
{
    switch (type) {
        case CYTOKINE_IL1B: return "IL-1β";
        case CYTOKINE_TNFA: return "TNF-α";
        case CYTOKINE_IL6: return "IL-6";
        case CYTOKINE_IL10: return "IL-10";
        case CYTOKINE_TGFB: return "TGF-β";
        default: return "UNKNOWN";
    }
}
