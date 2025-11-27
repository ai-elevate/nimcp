/**
 * @file nimcp_axon.c
 * @brief NIMCP Axon Module Implementation
 *
 * WHAT: Implementation of axon signal propagation and morphology
 * WHY:  Provide biologically-realistic action potential propagation
 * HOW:  Uses NIMCP utils for memory, threading, and math
 *
 * NIMCP STANDARDS:
 * - Functions < 50 lines
 * - Guard clauses for NULL checks
 * - nimcp_malloc/nimcp_free for memory
 */

#include "nimcp_axon.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

//=============================================================================
// INTERNAL CONSTANTS
//=============================================================================

/** Velocity coefficient for unmyelinated axons (m/s per sqrt(um)) */
static const float VELOCITY_COEFF_UNMYELINATED = 1.0f;

/** Velocity coefficient for myelinated axons (m/s per um diameter) */
static const float VELOCITY_COEFF_MYELINATED = 6.0f;

/** Activity EMA decay factor per millisecond */
static const float ACTIVITY_DECAY_FACTOR = 0.99f;

/** Minimum velocity to prevent division by zero */
static const float MIN_VELOCITY = 0.1f;

//=============================================================================
// SPIKE QUEUE INTERNAL STRUCTURE
//=============================================================================

struct axon_spike_queue_struct {
    axon_spike_event_t* events;
    uint32_t capacity;
    uint32_t count;
    uint32_t head;
    uint32_t tail;
    nimcp_mutex_t lock;
};

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

/**
 * @brief Calculate 3D distance
 */
float axon_distance_3d(const float a[3], const float b[3])
{
    if (!a || !b) return 0.0f;

    float dx = b[0] - a[0];
    float dy = b[1] - a[1];
    float dz = b[2] - a[2];

    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Validate axon parameters
 */
bool axon_validate_params(float length, float diameter)
{
    if (length <= 0.0f) return false;
    if (diameter < NIMCP_AXON_MIN_DIAMETER_UM) return false;
    if (diameter > NIMCP_AXON_MAX_DIAMETER_UM) return false;
    return true;
}

/**
 * @brief Clamp float to range
 */
static float clamp_f(float value, float min_val, float max_val)
{
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

//=============================================================================
// AXON CREATION AND DESTRUCTION
//=============================================================================

axon_t* axon_create(uint32_t id,
                    axon_type_t type,
                    uint32_t source_neuron_id,
                    uint32_t target_synapse_id,
                    float length,
                    float diameter)
{
    // Guard: validate parameters
    if (!axon_validate_params(length, diameter)) {
        return NULL;
    }

    // Allocate axon structure
    axon_t* axon = (axon_t*)nimcp_calloc(1, sizeof(axon_t));
    if (!axon) return NULL;

    // Initialize identification
    axon->id = id;
    axon->type = type;
    axon->state = AXON_STATE_RESTING;

    // Initialize connectivity
    axon->source_neuron_id = source_neuron_id;
    axon->target_synapse_id = target_synapse_id;
    axon->target_neuron_id = 0;  // Set by caller if needed

    // Initialize morphology
    axon->length = length;
    axon->diameter = clamp_f(diameter, NIMCP_AXON_MIN_DIAMETER_UM,
                             NIMCP_AXON_MAX_DIAMETER_UM);

    // No segments initially
    axon->num_segments = 0;
    axon->segments = NULL;

    // Initialize conduction based on type
    axon->myelination_level = (type == AXON_TYPE_MYELINATED ||
                               type == AXON_TYPE_A_ALPHA ||
                               type == AXON_TYPE_A_BETA) ? 0.5f : 0.0f;
    axon->mean_g_ratio = 0.77f;  // Optimal G-ratio

    // Calculate initial conduction properties
    axon_update_conduction(axon);

    // Initialize activity tracking
    memset(&axon->activity, 0, sizeof(axon_activity_stats_t));

    // Initialize metabolic state
    axon->atp_level = 1.0f;
    axon->lactate_level = 0.0f;

    // Initialize health
    axon->damage = 0.0f;
    axon->is_functional = true;

    // Initialize lock
    nimcp_mutex_init(&axon->lock, NULL);

    // Initialize memory pool (not enabled by default)
    axon->segment_pool = NULL;
    axon->use_segment_pool = false;

    // Initialize CoW (reference count starts at 1 for original)
    axon->cow_ref_count = 1;
    axon->cow_modified = false;
    axon->cow_original = NULL;

    return axon;
}

axon_t* axon_create_with_positions(uint32_t id,
                                    axon_type_t type,
                                    uint32_t source_neuron_id,
                                    uint32_t target_synapse_id,
                                    const float start_pos[3],
                                    const float end_pos[3],
                                    float diameter)
{
    // Guard clauses
    if (!start_pos || !end_pos) return NULL;

    // Calculate length from positions
    float length = axon_distance_3d(start_pos, end_pos);
    if (length <= 0.0f) length = 1.0f;  // Minimum length

    // Create base axon
    axon_t* axon = axon_create(id, type, source_neuron_id, target_synapse_id,
                                length, diameter);
    if (!axon) return NULL;

    // Set positions
    memcpy(axon->start_pos, start_pos, 3 * sizeof(float));
    memcpy(axon->end_pos, end_pos, 3 * sizeof(float));

    return axon;
}

void axon_destroy(axon_t* axon)
{
    if (!axon) return;

    // Check CoW reference count - only destroy if last reference
    if (axon->cow_ref_count > 1 && !axon->cow_original) {
        // This is an original with active CoW copies
        axon->cow_ref_count--;
        return;
    }

    // Destroy mutex
    nimcp_mutex_destroy(&axon->lock);

    // Free segments if allocated and owned (not a CoW sharing segments)
    if (axon->segments && (axon->cow_modified || !axon->cow_original)) {
        nimcp_free(axon->segments);
    }

    // Free segment pool if allocated
    if (axon->segment_pool) {
        axon_segment_pool_destroy(axon->segment_pool);
    }

    // Free axon structure
    nimcp_free(axon);
}

//=============================================================================
// AXON SEGMENTATION
//=============================================================================

bool axon_create_segments(axon_t* axon,
                          uint32_t num_segments,
                          float internode_length)
{
    // Guard clauses
    if (!axon) return false;
    if (num_segments == 0 || num_segments > NIMCP_AXON_MAX_SEGMENTS) return false;
    if (internode_length <= 0.0f) return false;

    // Free existing segments
    if (axon->segments) {
        nimcp_free(axon->segments);
    }

    // Allocate segment array
    axon->segments = (axon_segment_t*)nimcp_calloc(num_segments,
                                                    sizeof(axon_segment_t));
    if (!axon->segments) return false;

    axon->num_segments = num_segments;

    // Calculate direction vector
    float dir[3];
    for (int i = 0; i < 3; i++) {
        dir[i] = (axon->end_pos[i] - axon->start_pos[i]) / axon->length;
    }

    // Create alternating node/internode pattern
    float cumulative_length = 0.0f;
    float node_length = 1.0f;  // ~1 um for nodes of Ranvier

    for (uint32_t i = 0; i < num_segments; i++) {
        axon_segment_t* seg = &axon->segments[i];

        // Determine segment type
        if (i == 0) {
            seg->type = SEGMENT_TYPE_AIS;
            seg->length = 50.0f;  // Axon initial segment ~50 um
        } else if (i == num_segments - 1) {
            seg->type = SEGMENT_TYPE_TERMINAL;
            seg->length = 5.0f;  // Terminal bouton
        } else if (i % 2 == 1) {
            seg->type = SEGMENT_TYPE_NODE;
            seg->length = node_length;
        } else {
            seg->type = SEGMENT_TYPE_INTERNODE;
            seg->length = internode_length;
        }

        // Set position
        for (int j = 0; j < 3; j++) {
            seg->position[j] = axon->start_pos[j] + dir[j] * cumulative_length;
        }

        // Set diameter (slightly tapered)
        seg->diameter = axon->diameter * (1.0f - 0.1f * (float)i / num_segments);

        // Initialize myelination (internodes only)
        if (seg->type == SEGMENT_TYPE_INTERNODE) {
            seg->myelination = axon->myelination_level;
            seg->g_ratio = axon->mean_g_ratio;
        } else {
            seg->myelination = 0.0f;
            seg->g_ratio = 1.0f;  // No myelin
        }

        seg->oligo_id = 0;

        // Calculate local velocity
        if (seg->type == SEGMENT_TYPE_INTERNODE && seg->myelination > 0.0f) {
            seg->local_velocity = VELOCITY_COEFF_MYELINATED * seg->diameter *
                                  seg->myelination * NIMCP_AXON_MYELIN_MULTIPLIER / 50.0f;
        } else {
            seg->local_velocity = VELOCITY_COEFF_UNMYELINATED * sqrtf(seg->diameter);
        }
        if (seg->local_velocity < MIN_VELOCITY) {
            seg->local_velocity = MIN_VELOCITY;
        }

        // Calculate cumulative delay (ms) = length (um) / velocity (m/s) / 1000
        float segment_delay = seg->length / (seg->local_velocity * 1000.0f);
        if (i == 0) {
            seg->cumulative_delay = segment_delay;
        } else {
            seg->cumulative_delay = axon->segments[i - 1].cumulative_delay +
                                    segment_delay;
        }

        cumulative_length += seg->length;
    }

    // Update total propagation delay
    if (num_segments > 0) {
        axon->propagation_delay_ms = axon->segments[num_segments - 1].cumulative_delay;
    }

    return true;
}

bool axon_set_segment_myelination(axon_t* axon,
                                   uint32_t segment_index,
                                   float myelination,
                                   uint32_t oligo_id)
{
    // Guard clauses
    if (!axon) return false;
    if (!axon->segments) return false;
    if (segment_index >= axon->num_segments) return false;

    axon_segment_t* seg = &axon->segments[segment_index];

    // Only internodes can be myelinated
    if (seg->type != SEGMENT_TYPE_INTERNODE) return false;

    // Update myelination
    seg->myelination = clamp_f(myelination, 0.0f, 1.0f);
    seg->oligo_id = oligo_id;

    // Recalculate local velocity
    if (seg->myelination > 0.0f) {
        seg->local_velocity = VELOCITY_COEFF_MYELINATED * seg->diameter *
                              seg->myelination * NIMCP_AXON_MYELIN_MULTIPLIER / 50.0f;
    } else {
        seg->local_velocity = VELOCITY_COEFF_UNMYELINATED * sqrtf(seg->diameter);
    }
    if (seg->local_velocity < MIN_VELOCITY) {
        seg->local_velocity = MIN_VELOCITY;
    }

    // Recalculate cumulative delays from this segment onward
    for (uint32_t i = segment_index; i < axon->num_segments; i++) {
        axon_segment_t* s = &axon->segments[i];
        float segment_delay = s->length / (s->local_velocity * 1000.0f);

        if (i == 0) {
            s->cumulative_delay = segment_delay;
        } else {
            s->cumulative_delay = axon->segments[i - 1].cumulative_delay +
                                  segment_delay;
        }
    }

    // Update total delay
    axon->propagation_delay_ms = axon->segments[axon->num_segments - 1].cumulative_delay;

    // Update overall myelination level
    float total_myelin = 0.0f;
    uint32_t internode_count = 0;
    for (uint32_t i = 0; i < axon->num_segments; i++) {
        if (axon->segments[i].type == SEGMENT_TYPE_INTERNODE) {
            total_myelin += axon->segments[i].myelination;
            internode_count++;
        }
    }
    if (internode_count > 0) {
        axon->myelination_level = total_myelin / internode_count;
    }

    return true;
}

//=============================================================================
// CONDUCTION VELOCITY
//=============================================================================

float axon_calculate_velocity(const axon_t* axon)
{
    if (!axon) return MIN_VELOCITY;

    float velocity;

    // Type-specific velocity calculation
    switch (axon->type) {
        case AXON_TYPE_UNMYELINATED:
        case AXON_TYPE_C_FIBER:
            // Unmyelinated: v proportional to sqrt(diameter)
            velocity = VELOCITY_COEFF_UNMYELINATED * sqrtf(axon->diameter);
            break;

        case AXON_TYPE_MYELINATED:
        case AXON_TYPE_A_ALPHA:
        case AXON_TYPE_A_BETA:
        case AXON_TYPE_A_DELTA:
            // Myelinated: v proportional to diameter, scaled by myelination
            velocity = VELOCITY_COEFF_MYELINATED * axon->diameter *
                       (0.1f + 0.9f * axon->myelination_level);
            break;

        default:
            velocity = NIMCP_AXON_BASE_VELOCITY_MS;
    }

    // Clamp to valid range
    velocity = clamp_f(velocity, MIN_VELOCITY, NIMCP_AXON_MAX_VELOCITY_MS);

    // Apply damage penalty
    velocity *= (1.0f - axon->damage * 0.9f);

    return velocity;
}

void axon_update_conduction(axon_t* axon)
{
    if (!axon) return;

    // Calculate new velocity
    axon->effective_velocity = axon_calculate_velocity(axon);

    // Calculate propagation delay
    // delay (ms) = length (um) / velocity (m/s) / 1000
    axon->propagation_delay_ms = axon->length /
                                  (axon->effective_velocity * 1000.0f);

    // Update base velocity for reference
    axon->base_velocity = VELOCITY_COEFF_UNMYELINATED * sqrtf(axon->diameter);
}

//=============================================================================
// SPIKE PROPAGATION
//=============================================================================

bool axon_initiate_spike(axon_t* axon,
                         uint64_t current_time,
                         float amplitude)
{
    // Guard clauses
    if (!axon) return false;
    if (!axon->is_functional) return false;
    if (axon->damage >= 1.0f) return false;

    // Check refractory period
    if (axon_is_refractory(axon, current_time)) {
        return false;
    }

    // Check ATP level (metabolic gating)
    if (axon->atp_level < 0.1f) {
        return false;
    }

    // Acquire lock
    nimcp_mutex_lock(&axon->lock);

    // Update state
    axon->state = AXON_STATE_ACTIVE;

    // Set refractory period end
    uint64_t refractory_us = (uint64_t)(NIMCP_AXON_REFRACTORY_PERIOD_MS * 1000.0f);
    axon->refractory_end = current_time + refractory_us;

    // Update activity statistics
    axon->activity.total_spikes++;
    axon->activity.recent_spikes++;

    // Calculate inter-spike interval
    if (axon->activity.last_spike_time > 0) {
        float isi = (float)(current_time - axon->activity.last_spike_time) / 1000.0f;
        axon->activity.mean_isi = 0.9f * axon->activity.mean_isi + 0.1f * isi;
    }
    axon->activity.last_spike_time = current_time;

    // Consume ATP
    axon->atp_level -= 0.01f;
    if (axon->atp_level < 0.0f) axon->atp_level = 0.0f;

    nimcp_mutex_unlock(&axon->lock);

    return true;
}

bool axon_spike_arrived(axon_t* axon, uint64_t current_time)
{
    if (!axon) return false;
    if (axon->state != AXON_STATE_ACTIVE) return false;

    // Check if enough time has passed for spike to arrive
    uint64_t delay_us = (uint64_t)(axon->propagation_delay_ms * 1000.0f);
    uint64_t spike_time = axon->activity.last_spike_time;

    return (current_time >= spike_time + delay_us);
}

float axon_get_propagation_delay(const axon_t* axon)
{
    if (!axon) return 0.0f;
    return axon->propagation_delay_ms;
}

//=============================================================================
// MYELINATION INTERFACE
//=============================================================================

void axon_set_myelination(axon_t* axon, float myelination_level)
{
    if (!axon) return;

    axon->myelination_level = clamp_f(myelination_level, 0.0f, 1.0f);

    // Update all internode segments
    if (axon->segments) {
        for (uint32_t i = 0; i < axon->num_segments; i++) {
            if (axon->segments[i].type == SEGMENT_TYPE_INTERNODE) {
                axon->segments[i].myelination = axon->myelination_level;
            }
        }
    }

    // Recalculate conduction
    axon_update_conduction(axon);
}

float axon_get_myelination_signal(const axon_t* axon)
{
    if (!axon) return 0.0f;

    // Myelination signal based on activity
    // Higher activity = stronger signal for myelination
    float rate_factor = axon->activity.firing_rate / 50.0f;  // Normalize to 50 Hz
    rate_factor = clamp_f(rate_factor, 0.0f, 1.0f);

    // EMA provides smoothing
    float ema_factor = axon->activity.activity_ema;

    return 0.5f * rate_factor + 0.5f * ema_factor;
}

void axon_receive_lactate(axon_t* axon, float lactate_amount)
{
    if (!axon) return;
    if (lactate_amount <= 0.0f) return;

    nimcp_mutex_lock(&axon->lock);

    axon->lactate_level += lactate_amount;

    // Convert lactate to ATP (simplified metabolic model)
    float atp_gain = lactate_amount * 0.5f;  // 50% efficiency
    axon->atp_level += atp_gain;
    if (axon->atp_level > 1.0f) axon->atp_level = 1.0f;

    // Consume lactate
    axon->lactate_level *= 0.9f;

    nimcp_mutex_unlock(&axon->lock);
}

//=============================================================================
// ACTIVITY TRACKING
//=============================================================================

void axon_update_activity(axon_t* axon, uint64_t current_time)
{
    if (!axon) return;

    // Calculate time since last update (assume 1ms if unknown)
    float dt_ms = 1.0f;

    // Decay activity EMA
    float decay = powf(ACTIVITY_DECAY_FACTOR, dt_ms);
    axon->activity.activity_ema *= decay;

    // Add contribution from recent spikes
    if (axon->activity.recent_spikes > 0) {
        axon->activity.activity_ema += (float)axon->activity.recent_spikes * 0.1f;
        if (axon->activity.activity_ema > 1.0f) {
            axon->activity.activity_ema = 1.0f;
        }
        axon->activity.recent_spikes = 0;
    }

    // Calculate firing rate from mean ISI
    if (axon->activity.mean_isi > 0.0f) {
        axon->activity.firing_rate = 1000.0f / axon->activity.mean_isi;
    } else {
        axon->activity.firing_rate = 0.0f;
    }

    // Update myelination signal
    axon->activity.myelination_signal = axon_get_myelination_signal(axon);
}

float axon_get_firing_rate(const axon_t* axon)
{
    if (!axon) return 0.0f;
    return axon->activity.firing_rate;
}

void axon_get_activity_stats(const axon_t* axon, axon_activity_stats_t* stats)
{
    if (!stats) return;
    memset(stats, 0, sizeof(axon_activity_stats_t));
    if (!axon) return;

    memcpy(stats, &axon->activity, sizeof(axon_activity_stats_t));
}

//=============================================================================
// STATE MANAGEMENT
//=============================================================================

void axon_step(axon_t* axon, uint64_t current_time, float dt)
{
    if (!axon) return;
    if (!axon->is_functional) return;

    nimcp_mutex_lock(&axon->lock);

    // Update refractory state
    if (axon->state == AXON_STATE_REFRACTORY) {
        if (current_time >= axon->refractory_end) {
            axon->state = AXON_STATE_RESTING;
        }
    } else if (axon->state == AXON_STATE_ACTIVE) {
        // Transition to refractory after spike
        if (current_time >= axon->refractory_end) {
            axon->state = AXON_STATE_REFRACTORY;
        }
    }

    // Update activity tracking
    axon_update_activity(axon, current_time);

    // Regenerate ATP slowly
    axon->atp_level += 0.001f * dt;
    if (axon->atp_level > 1.0f) axon->atp_level = 1.0f;

    // Check for damage effects
    if (axon->damage >= 1.0f) {
        axon->is_functional = false;
        axon->state = AXON_STATE_DAMAGED;
    }

    nimcp_mutex_unlock(&axon->lock);
}

bool axon_is_refractory(const axon_t* axon, uint64_t current_time)
{
    if (!axon) return false;
    return (axon->state == AXON_STATE_REFRACTORY ||
            (axon->state == AXON_STATE_ACTIVE &&
             current_time < axon->refractory_end));
}

const char* axon_state_to_string(axon_state_t state)
{
    switch (state) {
        case AXON_STATE_RESTING:      return "Resting";
        case AXON_STATE_ACTIVE:       return "Active";
        case AXON_STATE_REFRACTORY:   return "Refractory";
        case AXON_STATE_DEMYELINATING: return "Demyelinating";
        case AXON_STATE_DAMAGED:      return "Damaged";
        default:                      return "Unknown";
    }
}

const char* axon_type_to_string(axon_type_t type)
{
    switch (type) {
        case AXON_TYPE_UNMYELINATED:  return "Unmyelinated";
        case AXON_TYPE_MYELINATED:    return "Myelinated";
        case AXON_TYPE_A_ALPHA:       return "A-alpha";
        case AXON_TYPE_A_BETA:        return "A-beta";
        case AXON_TYPE_A_DELTA:       return "A-delta";
        case AXON_TYPE_C_FIBER:       return "C-fiber";
        default:                      return "Unknown";
    }
}

//=============================================================================
// SPIKE QUEUE IMPLEMENTATION
//=============================================================================

axon_spike_queue_t* axon_spike_queue_create(uint32_t capacity)
{
    if (capacity == 0) capacity = NIMCP_AXON_SPIKE_QUEUE_SIZE;

    axon_spike_queue_t* queue = (axon_spike_queue_t*)nimcp_calloc(1,
                                                                   sizeof(axon_spike_queue_t));
    if (!queue) return NULL;

    queue->events = (axon_spike_event_t*)nimcp_calloc(capacity,
                                                       sizeof(axon_spike_event_t));
    if (!queue->events) {
        nimcp_free(queue);
        return NULL;
    }

    queue->capacity = capacity;
    queue->count = 0;
    queue->head = 0;
    queue->tail = 0;
    nimcp_mutex_init(&queue->lock, NULL);

    return queue;
}

void axon_spike_queue_destroy(axon_spike_queue_t* queue)
{
    if (!queue) return;

    nimcp_mutex_destroy(&queue->lock);

    if (queue->events) {
        nimcp_free(queue->events);
    }
    nimcp_free(queue);
}

bool axon_spike_queue_push(axon_spike_queue_t* queue,
                           const axon_spike_event_t* event)
{
    if (!queue || !event) return false;

    nimcp_mutex_lock(&queue->lock);

    if (queue->count >= queue->capacity) {
        nimcp_mutex_unlock(&queue->lock);
        return false;  // Queue full
    }

    // Add event at tail
    memcpy(&queue->events[queue->tail], event, sizeof(axon_spike_event_t));
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;

    nimcp_mutex_unlock(&queue->lock);

    return true;
}

bool axon_spike_queue_pop(axon_spike_queue_t* queue,
                          uint64_t current_time,
                          axon_spike_event_t* event)
{
    if (!queue || !event) return false;

    nimcp_mutex_lock(&queue->lock);

    if (queue->count == 0) {
        nimcp_mutex_unlock(&queue->lock);
        return false;
    }

    // Find event with arrival_time <= current_time
    // Simple linear scan (could optimize with heap)
    for (uint32_t i = 0; i < queue->count; i++) {
        uint32_t idx = (queue->head + i) % queue->capacity;
        if (queue->events[idx].arrival_time <= current_time) {
            // Copy event
            memcpy(event, &queue->events[idx], sizeof(axon_spike_event_t));

            // Remove by shifting (simple approach)
            // Move last element to this position
            if (i < queue->count - 1) {
                uint32_t last_idx = (queue->head + queue->count - 1) % queue->capacity;
                memcpy(&queue->events[idx], &queue->events[last_idx],
                       sizeof(axon_spike_event_t));
            }
            queue->count--;

            nimcp_mutex_unlock(&queue->lock);
            return true;
        }
    }

    nimcp_mutex_unlock(&queue->lock);
    return false;
}

uint32_t axon_spike_queue_size(const axon_spike_queue_t* queue)
{
    if (!queue) return 0;
    return queue->count;
}

//=============================================================================
// AXON NETWORK IMPLEMENTATION
//=============================================================================

axon_network_t* axon_network_create(uint32_t capacity)
{
    if (capacity == 0) capacity = 1000;
    if (capacity > NIMCP_AXON_NETWORK_MAX_AXONS) {
        capacity = NIMCP_AXON_NETWORK_MAX_AXONS;
    }

    axon_network_t* network = (axon_network_t*)nimcp_calloc(1,
                                                             sizeof(axon_network_t));
    if (!network) return NULL;

    network->axons = (axon_t**)nimcp_calloc(capacity, sizeof(axon_t*));
    if (!network->axons) {
        nimcp_free(network);
        return NULL;
    }

    network->spike_queue = axon_spike_queue_create(NIMCP_AXON_SPIKE_QUEUE_SIZE);
    if (!network->spike_queue) {
        nimcp_free(network->axons);
        nimcp_free(network);
        return NULL;
    }

    network->capacity = capacity;
    network->count = 0;
    network->current_time = 0;
    network->total_spikes_processed = 0;
    nimcp_mutex_init(&network->lock, NULL);
    network->spatial_index = NULL;
    network->spatial_index_valid = false;

    // Initialize spike pool (not enabled by default)
    network->spike_pool = NULL;
    network->use_spike_pool = false;

    return network;
}

void axon_network_destroy(axon_network_t* network)
{
    if (!network) return;

    // Destroy all axons
    if (network->axons) {
        for (uint32_t i = 0; i < network->count; i++) {
            if (network->axons[i]) {
                axon_destroy(network->axons[i]);
            }
        }
        nimcp_free(network->axons);
    }

    // Destroy spike queue
    if (network->spike_queue) {
        axon_spike_queue_destroy(network->spike_queue);
    }

    // Destroy spike pool
    if (network->spike_pool) {
        axon_spike_pool_destroy(network->spike_pool);
    }

    // Destroy network lock
    nimcp_mutex_destroy(&network->lock);

    nimcp_free(network);
}

bool axon_network_add(axon_network_t* network, axon_t* axon)
{
    if (!network || !axon) return false;

    nimcp_mutex_lock(&network->lock);

    if (network->count >= network->capacity) {
        nimcp_mutex_unlock(&network->lock);
        return false;
    }

    network->axons[network->count] = axon;
    network->count++;

    // Invalidate spatial index
    network->spatial_index_valid = false;

    nimcp_mutex_unlock(&network->lock);

    return true;
}

axon_t* axon_network_remove(axon_network_t* network, uint32_t axon_id)
{
    if (!network) return NULL;

    nimcp_mutex_lock(&network->lock);

    axon_t* removed = NULL;
    for (uint32_t i = 0; i < network->count; i++) {
        if (network->axons[i] && network->axons[i]->id == axon_id) {
            removed = network->axons[i];

            // Shift remaining axons
            for (uint32_t j = i; j < network->count - 1; j++) {
                network->axons[j] = network->axons[j + 1];
            }
            network->axons[network->count - 1] = NULL;
            network->count--;

            network->spatial_index_valid = false;
            break;
        }
    }

    nimcp_mutex_unlock(&network->lock);

    return removed;
}

axon_t* axon_network_find(axon_network_t* network, uint32_t axon_id)
{
    if (!network) return NULL;

    for (uint32_t i = 0; i < network->count; i++) {
        if (network->axons[i] && network->axons[i]->id == axon_id) {
            return network->axons[i];
        }
    }

    return NULL;
}

uint32_t axon_network_find_by_source(axon_network_t* network,
                                      uint32_t neuron_id,
                                      axon_t** results,
                                      uint32_t max_results)
{
    if (!network || !results || max_results == 0) return 0;

    uint32_t found = 0;
    for (uint32_t i = 0; i < network->count && found < max_results; i++) {
        if (network->axons[i] &&
            network->axons[i]->source_neuron_id == neuron_id) {
            results[found++] = network->axons[i];
        }
    }

    return found;
}

void axon_network_step(axon_network_t* network,
                       uint64_t current_time,
                       float dt)
{
    if (!network) return;

    nimcp_mutex_lock(&network->lock);

    network->current_time = current_time;

    // Step each axon
    for (uint32_t i = 0; i < network->count; i++) {
        if (network->axons[i]) {
            axon_step(network->axons[i], current_time, dt);
        }
    }

    nimcp_mutex_unlock(&network->lock);
}

uint32_t axon_network_process_arrivals(axon_network_t* network,
                                        uint64_t current_time,
                                        axon_spike_callback_t callback,
                                        void* user_data)
{
    if (!network || !network->spike_queue) return 0;

    uint32_t delivered = 0;
    axon_spike_event_t event;

    while (axon_spike_queue_pop(network->spike_queue, current_time, &event)) {
        // Find the axon
        axon_t* axon = axon_network_find(network, event.axon_id);

        if (axon && callback) {
            callback(axon, &event, user_data);
        }

        delivered++;
        network->total_spikes_processed++;
    }

    return delivered;
}

void axon_network_get_stats(const axon_network_t* network,
                            axon_network_stats_t* stats)
{
    if (!stats) return;
    memset(stats, 0, sizeof(axon_network_stats_t));
    if (!network) return;

    stats->total_axons = network->count;

    float total_myelination = 0.0f;
    float total_velocity = 0.0f;
    float total_delay = 0.0f;
    float total_rate = 0.0f;

    for (uint32_t i = 0; i < network->count; i++) {
        axon_t* axon = network->axons[i];
        if (!axon) continue;

        if (axon->myelination_level > 0.1f) {
            stats->myelinated_count++;
        } else {
            stats->unmyelinated_count++;
        }

        if (axon->state == AXON_STATE_DAMAGED) {
            stats->damaged_count++;
        }

        total_myelination += axon->myelination_level;
        total_velocity += axon->effective_velocity;
        total_delay += axon->propagation_delay_ms;
        total_rate += axon->activity.firing_rate;
    }

    if (network->count > 0) {
        stats->mean_myelination = total_myelination / network->count;
        stats->mean_velocity = total_velocity / network->count;
        stats->mean_delay = total_delay / network->count;
        stats->mean_firing_rate = total_rate / network->count;
    }

    stats->spikes_in_transit = axon_spike_queue_size(network->spike_queue);
}

//=============================================================================
// MEMORY POOL IMPLEMENTATION - SPIKE EVENTS
//=============================================================================

axon_spike_pool_t* axon_spike_pool_create(uint32_t capacity)
{
    if (capacity == 0) capacity = NIMCP_AXON_SPIKE_POOL_SIZE;

    axon_spike_pool_t* pool = (axon_spike_pool_t*)nimcp_calloc(1,
                                                                sizeof(axon_spike_pool_t));
    if (!pool) return NULL;

    // Allocate spike events array
    pool->events = (axon_spike_event_t*)nimcp_calloc(capacity,
                                                      sizeof(axon_spike_event_t));
    if (!pool->events) {
        nimcp_free(pool);
        return NULL;
    }

    // Calculate bitmap size
    pool->bitmap_words = (capacity + 63) / 64;
    pool->allocation_bitmap = (uint64_t*)nimcp_calloc(pool->bitmap_words,
                                                       sizeof(uint64_t));
    if (!pool->allocation_bitmap) {
        nimcp_free(pool->events);
        nimcp_free(pool);
        return NULL;
    }

    pool->capacity = capacity;
    pool->allocated_count = 0;
    pool->next_search_pos = 0;
    pool->high_water_mark = 0;

    return pool;
}

void axon_spike_pool_destroy(axon_spike_pool_t* pool)
{
    if (!pool) return;

    if (pool->allocation_bitmap) {
        nimcp_free(pool->allocation_bitmap);
    }
    if (pool->events) {
        nimcp_free(pool->events);
    }
    nimcp_free(pool);
}

axon_spike_event_t* axon_spike_pool_alloc(axon_spike_pool_t* pool)
{
    if (!pool) return NULL;
    if (pool->allocated_count >= pool->capacity) return NULL;

    // Search for free slot starting from next_search_pos
    uint32_t start_word = pool->next_search_pos / 64;

    for (uint32_t i = 0; i < pool->bitmap_words; i++) {
        uint32_t word_idx = (start_word + i) % pool->bitmap_words;
        uint64_t word = pool->allocation_bitmap[word_idx];

        // Check if any bit is free (0)
        if (word != UINT64_MAX) {
            // Find first zero bit
            uint32_t bit = 0;
            uint64_t mask = 1ULL;
            while (word & mask) {
                mask <<= 1;
                bit++;
            }

            uint32_t slot = word_idx * 64 + bit;
            if (slot >= pool->capacity) continue;

            // Mark as allocated
            pool->allocation_bitmap[word_idx] |= mask;
            pool->allocated_count++;

            // Update high water mark
            if (pool->allocated_count > pool->high_water_mark) {
                pool->high_water_mark = pool->allocated_count;
            }

            // Update search position
            pool->next_search_pos = slot + 1;

            return &pool->events[slot];
        }
    }

    return NULL;
}

void axon_spike_pool_free(axon_spike_pool_t* pool, axon_spike_event_t* event)
{
    if (!pool || !event) return;

    // Calculate slot index
    ptrdiff_t offset = event - pool->events;
    if (offset < 0 || (uint32_t)offset >= pool->capacity) return;

    uint32_t slot = (uint32_t)offset;
    uint32_t word_idx = slot / 64;
    uint32_t bit = slot % 64;

    // Check if actually allocated
    uint64_t mask = 1ULL << bit;
    if (!(pool->allocation_bitmap[word_idx] & mask)) return;

    // Mark as free
    pool->allocation_bitmap[word_idx] &= ~mask;
    pool->allocated_count--;

    // Optimize next search position
    if (slot < pool->next_search_pos) {
        pool->next_search_pos = slot;
    }
}

void axon_spike_pool_stats(const axon_spike_pool_t* pool,
                            uint32_t* allocated,
                            uint32_t* capacity,
                            uint32_t* high_water)
{
    if (!pool) {
        if (allocated) *allocated = 0;
        if (capacity) *capacity = 0;
        if (high_water) *high_water = 0;
        return;
    }

    if (allocated) *allocated = pool->allocated_count;
    if (capacity) *capacity = pool->capacity;
    if (high_water) *high_water = pool->high_water_mark;
}

//=============================================================================
// MEMORY POOL IMPLEMENTATION - SEGMENTS
//=============================================================================

axon_segment_pool_t* axon_segment_pool_create(uint32_t capacity)
{
    if (capacity == 0) capacity = NIMCP_AXON_SEGMENT_POOL_SIZE;

    axon_segment_pool_t* pool = (axon_segment_pool_t*)nimcp_calloc(1,
                                                                    sizeof(axon_segment_pool_t));
    if (!pool) return NULL;

    // Allocate segments array
    pool->segments = (axon_segment_t*)nimcp_calloc(capacity, sizeof(axon_segment_t));
    if (!pool->segments) {
        nimcp_free(pool);
        return NULL;
    }

    // Calculate bitmap size
    pool->bitmap_words = (capacity + 63) / 64;
    pool->allocation_bitmap = (uint64_t*)nimcp_calloc(pool->bitmap_words,
                                                       sizeof(uint64_t));
    if (!pool->allocation_bitmap) {
        nimcp_free(pool->segments);
        nimcp_free(pool);
        return NULL;
    }

    pool->capacity = capacity;
    pool->allocated_count = 0;
    pool->next_search_pos = 0;

    return pool;
}

void axon_segment_pool_destroy(axon_segment_pool_t* pool)
{
    if (!pool) return;

    if (pool->allocation_bitmap) {
        nimcp_free(pool->allocation_bitmap);
    }
    if (pool->segments) {
        nimcp_free(pool->segments);
    }
    nimcp_free(pool);
}

axon_segment_t* axon_segment_pool_alloc(axon_segment_pool_t* pool)
{
    if (!pool) return NULL;
    if (pool->allocated_count >= pool->capacity) return NULL;

    // Search for free slot
    uint32_t start_word = pool->next_search_pos / 64;

    for (uint32_t i = 0; i < pool->bitmap_words; i++) {
        uint32_t word_idx = (start_word + i) % pool->bitmap_words;
        uint64_t word = pool->allocation_bitmap[word_idx];

        if (word != UINT64_MAX) {
            // Find first zero bit
            uint32_t bit = 0;
            uint64_t mask = 1ULL;
            while (word & mask) {
                mask <<= 1;
                bit++;
            }

            uint32_t slot = word_idx * 64 + bit;
            if (slot >= pool->capacity) continue;

            // Mark as allocated
            pool->allocation_bitmap[word_idx] |= mask;
            pool->allocated_count++;
            pool->next_search_pos = slot + 1;

            // Clear segment data
            memset(&pool->segments[slot], 0, sizeof(axon_segment_t));

            return &pool->segments[slot];
        }
    }

    return NULL;
}

void axon_segment_pool_free(axon_segment_pool_t* pool, axon_segment_t* segment)
{
    if (!pool || !segment) return;

    // Calculate slot index
    ptrdiff_t offset = segment - pool->segments;
    if (offset < 0 || (uint32_t)offset >= pool->capacity) return;

    uint32_t slot = (uint32_t)offset;
    uint32_t word_idx = slot / 64;
    uint32_t bit = slot % 64;

    // Check if actually allocated
    uint64_t mask = 1ULL << bit;
    if (!(pool->allocation_bitmap[word_idx] & mask)) return;

    // Mark as free
    pool->allocation_bitmap[word_idx] &= ~mask;
    pool->allocated_count--;

    if (slot < pool->next_search_pos) {
        pool->next_search_pos = slot;
    }
}

//=============================================================================
// MEMORY POOL ENABLE FUNCTIONS
//=============================================================================

bool axon_network_enable_spike_pool(axon_network_t* network, uint32_t capacity)
{
    if (!network) return false;

    nimcp_mutex_lock(&network->lock);

    // Destroy existing pool if any
    if (network->spike_pool) {
        axon_spike_pool_destroy(network->spike_pool);
    }

    // Create new pool
    network->spike_pool = axon_spike_pool_create(capacity);
    network->use_spike_pool = (network->spike_pool != NULL);

    nimcp_mutex_unlock(&network->lock);

    return network->use_spike_pool;
}

bool axon_enable_segment_pool(axon_t* axon, uint32_t capacity)
{
    if (!axon) return false;

    nimcp_mutex_lock(&axon->lock);

    // Destroy existing pool if any
    if (axon->segment_pool) {
        axon_segment_pool_destroy(axon->segment_pool);
    }

    // Create new pool
    axon->segment_pool = axon_segment_pool_create(capacity);
    axon->use_segment_pool = (axon->segment_pool != NULL);

    nimcp_mutex_unlock(&axon->lock);

    return axon->use_segment_pool;
}

//=============================================================================
// COPY-ON-WRITE IMPLEMENTATION
//=============================================================================

axon_t* axon_cow_copy(axon_t* axon)
{
    if (!axon) return NULL;

    nimcp_mutex_lock(&axon->lock);

    // Allocate new axon structure (shallow copy)
    axon_t* copy = (axon_t*)nimcp_calloc(1, sizeof(axon_t));
    if (!copy) {
        nimcp_mutex_unlock(&axon->lock);
        return NULL;
    }

    // Copy scalar fields
    copy->id = axon->id;
    copy->type = axon->type;
    copy->state = axon->state;
    copy->source_neuron_id = axon->source_neuron_id;
    copy->target_synapse_id = axon->target_synapse_id;
    copy->target_neuron_id = axon->target_neuron_id;
    copy->length = axon->length;
    copy->diameter = axon->diameter;
    memcpy(copy->start_pos, axon->start_pos, sizeof(copy->start_pos));
    memcpy(copy->end_pos, axon->end_pos, sizeof(copy->end_pos));
    copy->base_velocity = axon->base_velocity;
    copy->effective_velocity = axon->effective_velocity;
    copy->propagation_delay_ms = axon->propagation_delay_ms;
    copy->myelination_level = axon->myelination_level;
    copy->mean_g_ratio = axon->mean_g_ratio;
    memcpy(&copy->activity, &axon->activity, sizeof(axon_activity_stats_t));
    copy->refractory_end = axon->refractory_end;
    copy->atp_level = axon->atp_level;
    copy->lactate_level = axon->lactate_level;
    copy->damage = axon->damage;
    copy->is_functional = axon->is_functional;

    // Share segments (CoW - don't copy yet)
    copy->segments = axon->segments;
    copy->num_segments = axon->num_segments;

    // Initialize own mutex
    nimcp_mutex_init(&copy->lock, NULL);

    // CoW bookkeeping
    copy->segment_pool = NULL;  // Don't share pool
    copy->use_segment_pool = false;
    copy->cow_ref_count = 1;
    copy->cow_modified = false;
    copy->cow_original = axon;

    // Increment original's reference count
    axon->cow_ref_count++;

    nimcp_mutex_unlock(&axon->lock);

    return copy;
}

bool axon_cow_prepare_write(axon_t* axon)
{
    if (!axon) return false;

    nimcp_mutex_lock(&axon->lock);

    // If already modified or no original, we own the data
    if (axon->cow_modified || !axon->cow_original) {
        axon->cow_modified = true;
        nimcp_mutex_unlock(&axon->lock);
        return true;
    }

    // Need to deep copy shared data
    if (axon->segments && axon->num_segments > 0) {
        axon_segment_t* old_segments = axon->segments;

        // Allocate our own copy
        axon->segments = (axon_segment_t*)nimcp_calloc(axon->num_segments,
                                                        sizeof(axon_segment_t));
        if (!axon->segments) {
            axon->segments = old_segments;  // Restore
            nimcp_mutex_unlock(&axon->lock);
            return false;
        }

        // Copy segment data
        memcpy(axon->segments, old_segments,
               axon->num_segments * sizeof(axon_segment_t));

        // Decrement original's reference count
        if (axon->cow_original) {
            nimcp_mutex_lock(&axon->cow_original->lock);
            if (axon->cow_original->cow_ref_count > 0) {
                axon->cow_original->cow_ref_count--;
            }
            nimcp_mutex_unlock(&axon->cow_original->lock);
        }
    }

    axon->cow_original = NULL;
    axon->cow_modified = true;

    nimcp_mutex_unlock(&axon->lock);

    return true;
}

void axon_cow_release(axon_t* axon)
{
    if (!axon) return;

    nimcp_mutex_lock(&axon->lock);

    // Decrement reference count
    if (axon->cow_ref_count > 0) {
        axon->cow_ref_count--;
    }

    // If this is a CoW copy, decrement original's count
    if (axon->cow_original) {
        nimcp_mutex_lock(&axon->cow_original->lock);
        if (axon->cow_original->cow_ref_count > 0) {
            axon->cow_original->cow_ref_count--;
        }
        nimcp_mutex_unlock(&axon->cow_original->lock);
    }

    nimcp_mutex_unlock(&axon->lock);

    // If reference count is 0 and this is a CoW copy, free
    if (axon->cow_ref_count == 0 && axon->cow_original) {
        // Only free segments if we own them (modified)
        if (axon->cow_modified && axon->segments) {
            nimcp_free(axon->segments);
        }
        if (axon->segment_pool) {
            axon_segment_pool_destroy(axon->segment_pool);
        }
        nimcp_mutex_destroy(&axon->lock);
        nimcp_free(axon);
    }
}

bool axon_is_cow_copy(const axon_t* axon)
{
    if (!axon) return false;
    return (axon->cow_original != NULL);
}

uint32_t axon_cow_ref_count(const axon_t* axon)
{
    if (!axon) return 0;
    return axon->cow_ref_count;
}

//=============================================================================
// ENHANCED BIOPHYSICS IMPLEMENTATION (from nimcp_myelin_math.h integration)
//=============================================================================

bool axon_init_biophysics(axon_t* axon, bool use_stochastic, uint64_t seed)
{
    if (!axon) return false;

    nimcp_mutex_lock(&axon->lock);

    // Create biophysics state
    axon->biophysics = nimcp_myelin_biophysics_create(use_stochastic, seed);
    if (!axon->biophysics) {
        nimcp_mutex_unlock(&axon->lock);
        return false;
    }

    // Initialize temperature to normal body temperature
    axon->temperature_c = 37.0f;

    nimcp_mutex_unlock(&axon->lock);

    // Update all segments with enhanced calculations
    axon_update_biophysics(axon);

    return true;
}

void axon_update_segment_cable_params(axon_t* axon, uint32_t segment_index)
{
    if (!axon || !axon->segments || segment_index >= axon->num_segments) return;

    axon_segment_t* seg = &axon->segments[segment_index];

    // Only calculate for myelinated internodes
    if (seg->type == SEGMENT_TYPE_INTERNODE) {
        // Estimate lamellae from myelination level
        uint32_t lamellae = (uint32_t)(seg->myelination * 40.0f);
        if (lamellae < 1) lamellae = 1;

        nimcp_myelin_compute_cable_params(seg->diameter, lamellae, &seg->cable_params);

        // Compute optimal g-ratio for this diameter
        seg->optimal_g_ratio = nimcp_myelin_optimal_g_ratio(seg->diameter);
    }
}

float axon_compute_segment_velocity_enhanced(axon_t* axon, uint32_t segment_index)
{
    if (!axon || !axon->segments || segment_index >= axon->num_segments) {
        return NIMCP_AXON_BASE_VELOCITY_MS;
    }

    axon_segment_t* seg = &axon->segments[segment_index];

    // For non-internode segments, use basic velocity
    if (seg->type != SEGMENT_TYPE_INTERNODE) {
        seg->local_velocity = NIMCP_AXON_BASE_VELOCITY_MS;
        seg->is_conducting = true;
        seg->block_probability = 0.0f;
        return seg->local_velocity;
    }

    // Estimate lamellae and compaction from myelination level
    uint32_t lamellae = (uint32_t)(seg->myelination * 40.0f);
    if (lamellae < 1) lamellae = 1;
    float compaction = seg->myelination;
    float integrity = 1.0f - (axon->damage);

    // Use enhanced saltatory conduction model
    float velocity = nimcp_myelin_saltatory_velocity(
        seg->diameter,
        seg->length,
        lamellae,
        seg->g_ratio,
        compaction,
        integrity,
        &seg->saltatory
    );

    // Store block probability
    seg->block_probability = seg->saltatory.block_probability;
    seg->is_conducting = !seg->saltatory.is_blocked;
    seg->local_velocity = velocity;

    return velocity;
}

bool axon_check_segment_block(axon_t* axon, uint32_t segment_index)
{
    if (!axon || !axon->segments || segment_index >= axon->num_segments) {
        return true;  // Block if invalid
    }

    axon_segment_t* seg = &axon->segments[segment_index];
    float integrity = 1.0f - axon->damage;

    nimcp_conduction_block_params_t params = nimcp_myelin_block_params_default();
    seg->block_probability = nimcp_myelin_block_probability(
        integrity, axon->temperature_c, &params);

    seg->is_conducting = (seg->block_probability < 0.5f);
    return !seg->is_conducting;
}

void axon_set_temperature(axon_t* axon, float temperature_c)
{
    if (!axon) return;

    axon->temperature_c = clamp_f(temperature_c, 20.0f, 45.0f);

    if (axon->biophysics) {
        axon->biophysics->temperature_c = axon->temperature_c;
    }
}

void axon_compute_metabolic_efficiency(axon_t* axon)
{
    if (!axon || axon->num_segments == 0) return;

    // Calculate mean compaction and integrity
    float mean_compaction = 0.0f;
    float integrity = 1.0f - axon->damage;
    uint32_t myelinated_count = 0;

    for (uint32_t i = 0; i < axon->num_segments; i++) {
        if (axon->segments[i].type == SEGMENT_TYPE_INTERNODE) {
            mean_compaction += axon->segments[i].myelination;
            myelinated_count++;
        }
    }

    if (myelinated_count > 0) {
        mean_compaction /= (float)myelinated_count;
    }

    // Compute metabolic efficiency
    nimcp_myelin_compute_metabolic_efficiency(
        axon->length,
        axon->diameter,
        myelinated_count,
        mean_compaction,
        integrity,
        &axon->metabolic_efficiency
    );
}

float axon_get_atp_per_ap(const axon_t* axon)
{
    if (!axon) return 0.0f;
    return axon->metabolic_efficiency.atp_per_ap;
}

float axon_get_efficiency_ratio(const axon_t* axon)
{
    if (!axon) return 1.0f;
    return axon->metabolic_efficiency.efficiency_ratio;
}

void axon_update_biophysics(axon_t* axon)
{
    if (!axon) return;

    nimcp_mutex_lock(&axon->lock);

    float total_lambda = 0.0f;
    float total_block_prob = 0.0f;
    uint32_t internode_count = 0;

    // Update each segment
    for (uint32_t i = 0; i < axon->num_segments; i++) {
        axon_segment_t* seg = &axon->segments[i];

        // Update cable parameters
        axon_update_segment_cable_params(axon, i);

        // Update velocity with enhanced model
        axon_compute_segment_velocity_enhanced(axon, i);

        // Update block probability
        axon_check_segment_block(axon, i);

        if (seg->type == SEGMENT_TYPE_INTERNODE) {
            total_lambda += seg->cable_params.lambda_um;
            total_block_prob += seg->block_probability;
            internode_count++;
        }
    }

    // Update aggregate values
    if (internode_count > 0) {
        axon->mean_lambda_um = total_lambda / (float)internode_count;
        axon->overall_block_probability = total_block_prob / (float)internode_count;
    }

    nimcp_mutex_unlock(&axon->lock);

    // Update conduction properties
    axon_update_conduction(axon);

    // Update metabolic efficiency
    axon_compute_metabolic_efficiency(axon);
}

float axon_apply_activity_myelination(axon_t* axon, float activity, float dt)
{
    if (!axon || dt <= 0.0f) return 0.0f;

    nimcp_myelination_kinetics_t kinetics = nimcp_myelin_kinetics_default();
    float total_delta = 0.0f;

    nimcp_mutex_lock(&axon->lock);

    for (uint32_t i = 0; i < axon->num_segments; i++) {
        axon_segment_t* seg = &axon->segments[i];

        if (seg->type == SEGMENT_TYPE_INTERNODE) {
            // Estimate current lamellae from myelination level
            float current_lamellae = seg->myelination * 40.0f;

            // Compute myelination rate using Hill kinetics
            float rate = nimcp_myelin_compute_myelination_rate(
                activity, current_lamellae, &kinetics);

            float new_lamellae = nimcp_myelin_update_lamellae(
                current_lamellae, activity, dt, &kinetics);

            float delta = new_lamellae - current_lamellae;
            total_delta += delta;

            // Update myelination level
            seg->myelination = clamp_f(new_lamellae / 40.0f, 0.0f, 1.0f);
        }
    }

    // Update overall myelination level
    float total_myelination = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < axon->num_segments; i++) {
        if (axon->segments[i].type == SEGMENT_TYPE_INTERNODE) {
            total_myelination += axon->segments[i].myelination;
            count++;
        }
    }
    if (count > 0) {
        axon->myelination_level = total_myelination / (float)count;
    }

    nimcp_mutex_unlock(&axon->lock);

    // Recalculate after changes
    axon_update_biophysics(axon);

    return total_delta;
}

float axon_get_optimal_g_ratio(const axon_t* axon)
{
    if (!axon) return 0.77f;
    return nimcp_myelin_optimal_g_ratio(axon->diameter);
}

float axon_get_space_constant(const axon_t* axon)
{
    if (!axon) return 0.0f;
    return axon->mean_lambda_um;
}

float axon_get_frequency_threshold(const axon_t* axon, float frequency_hz)
{
    if (!axon) return 1.0f;

    nimcp_conduction_block_params_t params = nimcp_myelin_block_params_default();
    return nimcp_myelin_frequency_threshold(frequency_hz, axon->temperature_c, &params);
}

void axon_apply_myelination_variability(axon_t* axon)
{
    if (!axon || !axon->biophysics || !axon->biophysics->use_stochastic) return;

    nimcp_mutex_lock(&axon->lock);

    nimcp_myelin_rng_t* rng = &axon->biophysics->rng;

    for (uint32_t i = 0; i < axon->num_segments; i++) {
        axon_segment_t* seg = &axon->segments[i];

        if (seg->type == SEGMENT_TYPE_INTERNODE) {
            // Apply variability to myelination level
            float target_myelination = seg->myelination;
            float varied = nimcp_myelin_rng_normal(rng, target_myelination, 0.05f);
            seg->myelination = clamp_f(varied, 0.0f, 1.0f);

            // Apply variability to g-ratio
            float varied_g = nimcp_myelin_vary_g_ratio(rng, seg->g_ratio);
            seg->g_ratio = varied_g;

            // Apply variability to segment length
            seg->length = nimcp_myelin_vary_internode(rng, seg->length);
        }
    }

    nimcp_mutex_unlock(&axon->lock);

    // Recalculate after variability applied
    axon_update_biophysics(axon);
}
