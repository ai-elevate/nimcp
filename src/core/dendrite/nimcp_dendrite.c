/**
 * @file nimcp_dendrite.c
 * @brief Dendrite module implementation with biological accuracy
 *
 * WHAT: Complete dendrite implementation with segments, spines, and dynamics
 * WHY:  Enable realistic dendritic computation, compartmentalization, and plasticity
 * HOW:  Cable theory, spine morphology, calcium dynamics, LTP/LTD integration
 *
 * FEATURES:
 * - Dendritic segments with cable properties
 * - Dendritic spines (thin, stubby, mushroom, filopodia)
 * - Calcium dynamics and plasticity
 * - Structural plasticity (spine formation/elimination)
 * - Attenuation and integration windows
 * - Thread-safe operations
 *
 * INTEGRATION:
 * - Neurons: Multiple dendrites per neuron
 * - Synapses: Connect to specific spines
 * - Axons: Complete signal propagation chain
 *
 * @version Phase: Dendrite Integration
 * @date 2025-11-24
 */

#include "core/dendrite/nimcp_dendrite.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>
#include <pthread.h>

// ============================================================================
// Constants
// ============================================================================

// Biological constants
#define DENDRITE_DEFAULT_DIAMETER 2.0f      // μm
#define DENDRITE_DEFAULT_LENGTH 100.0f      // μm
#define DENDRITE_R_M 20000.0f              // Ω·cm² (membrane resistance)
#define DENDRITE_C_M 1.0f                   // μF/cm² (membrane capacitance)
#define DENDRITE_R_A 150.0f                 // Ω·cm (axial resistance)

// Spine constants
#define SPINE_NECK_LENGTH_THIN 1.0f         // μm
#define SPINE_NECK_LENGTH_STUBBY 0.2f       // μm
#define SPINE_NECK_LENGTH_MUSHROOM 0.8f     // μm
#define SPINE_NECK_LENGTH_FILOPODIA 2.0f    // μm

#define SPINE_HEAD_DIAMETER_THIN 0.3f       // μm
#define SPINE_HEAD_DIAMETER_STUBBY 0.6f     // μm
#define SPINE_HEAD_DIAMETER_MUSHROOM 0.8f   // μm
#define SPINE_HEAD_DIAMETER_FILOPODIA 0.2f  // μm

// Plasticity constants
#define LTP_THRESHOLD_DEFAULT 0.8f          // mV (calcium threshold for LTP)
#define LTD_THRESHOLD_DEFAULT 0.3f          // mV (calcium threshold for LTD)
#define STRUCTURAL_PLASTICITY_RATE 0.001f   // Per timestep

// Calcium dynamics
#define CALCIUM_DECAY_TAU 20.0f             // ms
#define CALCIUM_BUFFER_CAPACITY 100.0f      // Buffering capacity

// ============================================================================
// Helper Function Prototypes
// ============================================================================

static dendritic_segment_t* create_segment(
    uint32_t id,
    dendrite_segment_type_t type,
    uint32_t parent_segment,
    float position[3],
    float length,
    float diameter
);

static dendritic_spine_t* create_spine(
    uint32_t id,
    spine_type_t type,
    uint32_t dendrite_id,
    uint32_t segment_id
);

static void calculate_cable_properties(dendritic_segment_t* segment);
static void update_segment_voltage(dendritic_segment_t* segment, float dt_ms);
static void update_segment_calcium(dendritic_segment_t* segment, float dt_ms);
static float calculate_attenuation(dendrite_t* dendrite, uint32_t segment_id);

// ============================================================================
// Dendrite Creation and Destruction
// ============================================================================

/**
 * WHAT: Create dendrite with specified configuration
 * WHY:  Initialize dendrite structure with biological properties
 * HOW:  Allocate memory, set parameters, initialize mutex
 */
dendrite_t* dendrite_create(dendrite_config_t* config) {
    // Guard: NULL config
    if (!config) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_create: NULL config");
        return NULL;
    }

    // Allocate dendrite
    dendrite_t* dendrite = (dendrite_t*)nimcp_calloc(1, sizeof(dendrite_t));
    if (!dendrite) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_create: Failed to allocate dendrite");
        return NULL;
    }

    // Initialize basic properties
    dendrite->id = config->id;
    dendrite->type = config->type;
    dendrite->state = DENDRITE_STATE_NORMAL;
    dendrite->target_neuron_id = config->target_neuron_id;

    // Initialize morphology
    dendrite->total_length = config->total_length > 0 ? config->total_length : DENDRITE_DEFAULT_LENGTH;
    dendrite->mean_diameter = config->mean_diameter > 0 ? config->mean_diameter : DENDRITE_DEFAULT_DIAMETER;

    memcpy(dendrite->start_pos, config->start_pos, sizeof(float) * 3);

    // Allocate segments array
    dendrite->num_segments = 0;
    dendrite->segments = NULL;  // Will be allocated when creating segments

    // Allocate spines array
    dendrite->num_spines = 0;
    dendrite->spines = NULL;  // Will be allocated when adding spines

    // Initialize electrical properties (will be calculated after adding segments)
    dendrite->input_resistance = 0.0f;
    dendrite->time_constant = 0.0f;
    dendrite->attenuation_factor = 0.0f;
    dendrite->integration_window_ms = config->integration_window_ms > 0 ?
                                      config->integration_window_ms : 20.0f;

    // Initialize voltages
    dendrite->somatic_voltage = 0.0f;
    dendrite->mean_voltage = 0.0f;
    dendrite->calcium_level = 0.0f;

    // Initialize activity stats
    memset(&dendrite->activity, 0, sizeof(dendrite_activity_stats_t));

    // Initialize plasticity
    dendrite->structural_plasticity = config->structural_plasticity;
    dendrite->ltp_threshold = config->ltp_threshold > 0 ?
                              config->ltp_threshold : LTP_THRESHOLD_DEFAULT;
    dendrite->ltd_threshold = config->ltd_threshold > 0 ?
                              config->ltd_threshold : LTD_THRESHOLD_DEFAULT;

    // Initialize health
    dendrite->atp_level = 1.0f;
    dendrite->damage = 0.0f;
    dendrite->is_functional = true;

    // Initialize mutex
    pthread_mutex_init(&dendrite->lock, NULL);

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_create: Created dendrite %u (type=%d)",
              dendrite->id, dendrite->type);

    return dendrite;
}

/**
 * WHAT: Destroy dendrite and free all resources
 * WHY:  Clean up memory when dendrite no longer needed
 * HOW:  Free segments, spines, destroy mutex
 */
void dendrite_destroy(dendrite_t* dendrite) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return;
    }

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_destroy: Destroying dendrite %u", dendrite->id);

    // Free segments
    if (dendrite->segments) {
        nimcp_free(dendrite->segments);
        dendrite->segments = NULL;
    }

    // Free spines
    if (dendrite->spines) {
        nimcp_free(dendrite->spines);
        dendrite->spines = NULL;
    }

    // Destroy mutex
    pthread_mutex_destroy(&dendrite->lock);

    // Free dendrite
    nimcp_free(dendrite);
}

// ============================================================================
// Segment Management
// ============================================================================

/**
 * WHAT: Create dendritic segments for the dendrite
 * WHY:  Build compartmental structure for cable theory
 * HOW:  Allocate segment array, set up parent-child relationships
 */
bool dendrite_create_segments(
    dendrite_t* dendrite,
    uint32_t num_segments,
    segment_config_t* segment_configs
) {
    // Guard: NULL inputs
    if (!dendrite || !segment_configs || num_segments == 0) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_create_segments: Invalid inputs");
        return false;
    }

    // Guard: Already has segments
    if (dendrite->segments) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_create_segments: Dendrite %u already has segments",
                  dendrite->id);
        return false;
    }

    // Allocate segments
    dendrite->segments = (dendritic_segment_t*)nimcp_calloc(
        num_segments,
        sizeof(dendritic_segment_t)
    );
    if (!dendrite->segments) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_create_segments: Failed to allocate segments");
        return false;
    }

    // Create each segment
    for (uint32_t i = 0; i < num_segments; i++) {
        segment_config_t* config = &segment_configs[i];

        dendritic_segment_t* segment = &dendrite->segments[i];
        segment->id = i;
        segment->type = config->type;
        segment->parent_segment = config->parent_segment;
        segment->num_children = 0;

        memcpy(segment->position, config->position, sizeof(float) * 3);
        segment->length = config->length;
        segment->diameter = config->diameter;
        segment->path_distance = config->path_distance;

        // Calculate cable properties
        calculate_cable_properties(segment);

        // Initialize electrical state
        segment->voltage = 0.0f;
        segment->calcium = 0.0f;

        // Initialize spines
        segment->num_spines = 0;
        memset(segment->spine_ids, 0, sizeof(segment->spine_ids));

        // Active properties
        segment->has_active_properties = config->has_active_properties;
        if (segment->has_active_properties) {
            segment->g_na = 120.0f;  // mS/cm²
            segment->g_k = 36.0f;    // mS/cm²
            segment->g_ca = 1.0f;    // mS/cm²
        } else {
            segment->g_na = 0.0f;
            segment->g_k = 0.0f;
            segment->g_ca = 0.0f;
        }
    }

    dendrite->num_segments = num_segments;

    // Build parent-child relationships
    for (uint32_t i = 0; i < num_segments; i++) {
        dendritic_segment_t* segment = &dendrite->segments[i];
        if (segment->parent_segment != UINT32_MAX) {
            dendritic_segment_t* parent = &dendrite->segments[segment->parent_segment];
            if (parent->num_children < NIMCP_DENDRITE_MAX_CHILDREN) {
                parent->child_segments[parent->num_children++] = i;
            }
        }
    }

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_create_segments: Created %u segments for dendrite %u",
              num_segments, dendrite->id);

    return true;
}

/**
 * WHAT: Add branch to dendrite at specified segment
 * WHY:  Support dendritic tree morphology
 * HOW:  Create new segment, attach to parent
 */
uint32_t dendrite_add_branch(
    dendrite_t* dendrite,
    uint32_t parent_segment_id,
    float length,
    float diameter,
    float branch_angle
) {
    // Guard: NULL dendrite
    if (!dendrite) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_branch: NULL dendrite");
        return UINT32_MAX;
    }

    // Guard: Invalid parent
    if (parent_segment_id >= dendrite->num_segments) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_branch: Invalid parent segment %u",
                  parent_segment_id);
        return UINT32_MAX;
    }

    dendritic_segment_t* parent = &dendrite->segments[parent_segment_id];

    // Guard: Parent has too many children
    if (parent->num_children >= NIMCP_DENDRITE_MAX_CHILDREN) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_branch: Parent segment %u has max children",
                  parent_segment_id);
        return UINT32_MAX;
    }

    // Reallocate segments array
    uint32_t new_id = dendrite->num_segments;
    dendrite->num_segments++;

    dendrite->segments = (dendritic_segment_t*)nimcp_realloc(
        dendrite->segments,
        dendrite->num_segments * sizeof(dendritic_segment_t)
    );
    if (!dendrite->segments) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_branch: Failed to reallocate segments");
        return UINT32_MAX;
    }

    // Re-get parent pointer after realloc (memory may have moved)
    parent = &dendrite->segments[parent_segment_id];

    // Initialize new segment
    dendritic_segment_t* new_segment = &dendrite->segments[new_id];
    memset(new_segment, 0, sizeof(dendritic_segment_t));

    new_segment->id = new_id;
    new_segment->type = DENDRITE_SEGMENT_OBLIQUE;
    new_segment->parent_segment = parent_segment_id;
    new_segment->num_children = 0;

    // Calculate position based on parent and branch angle
    float dx = cosf(branch_angle) * length;
    float dy = sinf(branch_angle) * length;
    new_segment->position[0] = parent->position[0] + dx;
    new_segment->position[1] = parent->position[1] + dy;
    new_segment->position[2] = parent->position[2];

    new_segment->length = length;
    new_segment->diameter = diameter;
    new_segment->path_distance = parent->path_distance + length;

    // Calculate cable properties
    calculate_cable_properties(new_segment);

    // Add to parent's children
    parent->child_segments[parent->num_children++] = new_id;

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_add_branch: Added branch segment %u to dendrite %u",
              new_id, dendrite->id);

    return new_id;
}

// ============================================================================
// Spine Management
// ============================================================================

/**
 * WHAT: Add dendritic spine to segment
 * WHY:  Enable synaptic connections with morphological detail
 * HOW:  Create spine structure, link to segment
 */
uint32_t dendrite_add_spine(
    dendrite_t* dendrite,
    uint32_t segment_id,
    spine_type_t type,
    uint32_t synapse_id
) {
    // Guard: NULL dendrite
    if (!dendrite) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_spine: NULL dendrite");
        return UINT32_MAX;
    }

    // Guard: Invalid segment
    if (segment_id >= dendrite->num_segments) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_spine: Invalid segment %u", segment_id);
        return UINT32_MAX;
    }

    dendritic_segment_t* segment = &dendrite->segments[segment_id];

    // Guard: Segment has too many spines
    if (segment->num_spines >= 64) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_spine: Segment %u has max spines", segment_id);
        return UINT32_MAX;
    }

    // Reallocate spines array
    uint32_t new_id = dendrite->num_spines;
    dendrite->num_spines++;

    dendrite->spines = (dendritic_spine_t*)nimcp_realloc(
        dendrite->spines,
        dendrite->num_spines * sizeof(dendritic_spine_t)
    );
    if (!dendrite->spines) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_add_spine: Failed to reallocate spines");
        return UINT32_MAX;
    }

    // Initialize new spine
    dendritic_spine_t* spine = &dendrite->spines[new_id];
    memset(spine, 0, sizeof(dendritic_spine_t));

    spine->id = new_id;
    spine->type = type;
    spine->state = SPINE_STATE_STABLE;
    spine->dendrite_id = dendrite->id;
    spine->segment_id = segment_id;
    spine->synapse_id = synapse_id;

    // Set morphology based on type
    switch (type) {
        case SPINE_TYPE_THIN:
            spine->neck_length = SPINE_NECK_LENGTH_THIN;
            spine->neck_diameter = 0.1f;
            spine->head_diameter = SPINE_HEAD_DIAMETER_THIN;
            break;
        case SPINE_TYPE_STUBBY:
            spine->neck_length = SPINE_NECK_LENGTH_STUBBY;
            spine->neck_diameter = 0.3f;
            spine->head_diameter = SPINE_HEAD_DIAMETER_STUBBY;
            break;
        case SPINE_TYPE_MUSHROOM:
            spine->neck_length = SPINE_NECK_LENGTH_MUSHROOM;
            spine->neck_diameter = 0.15f;
            spine->head_diameter = SPINE_HEAD_DIAMETER_MUSHROOM;
            break;
        case SPINE_TYPE_FILOPODIA:
            spine->neck_length = SPINE_NECK_LENGTH_FILOPODIA;
            spine->neck_diameter = 0.05f;
            spine->head_diameter = SPINE_HEAD_DIAMETER_FILOPODIA;
            break;
        default:
            spine->neck_length = 1.0f;
            spine->neck_diameter = 0.1f;
            spine->head_diameter = 0.3f;
            break;
    }

    // Calculate head volume (sphere approximation)
    float radius = spine->head_diameter / 2.0f;
    spine->head_volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;

    // Calculate electrical properties
    spine->neck_resistance = (4.0f * DENDRITE_R_A * spine->neck_length) /
                             (M_PI * spine->neck_diameter * spine->neck_diameter);
    spine->head_capacitance = DENDRITE_C_M * 4.0f * M_PI * radius * radius;

    // Initialize state
    spine->voltage = 0.0f;
    spine->calcium = 0.0f;

    // Initialize synaptic properties
    spine->synaptic_weight = 1.0f;
    spine->ampa_receptors = 100.0f;
    spine->nmda_receptors = 20.0f;

    // Initialize plasticity
    spine->growth_factor = 1.0f;
    spine->stability = 0.5f;
    spine->creation_time = 0;  // Will be set by caller

    // Add to segment's spine list
    segment->spine_ids[segment->num_spines++] = new_id;

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_add_spine: Added spine %u (type=%d) to segment %u",
              new_id, type, segment_id);

    return new_id;
}

/**
 * WHAT: Remove dendritic spine
 * WHY:  Support structural plasticity and spine elimination
 * HOW:  Remove from segment list, mark as inactive
 */
bool dendrite_remove_spine(dendrite_t* dendrite, uint32_t spine_id) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return false;
    }

    // Guard: Invalid spine ID
    if (spine_id >= dendrite->num_spines) {
        return false;
    }

    dendritic_spine_t* spine = &dendrite->spines[spine_id];
    uint32_t segment_id = spine->segment_id;

    // Guard: Invalid segment
    if (segment_id >= dendrite->num_segments) {
        return false;
    }

    dendritic_segment_t* segment = &dendrite->segments[segment_id];

    // Remove from segment's spine list
    for (uint32_t i = 0; i < segment->num_spines; i++) {
        if (segment->spine_ids[i] == spine_id) {
            // Shift remaining spines
            for (uint32_t j = i; j < segment->num_spines - 1; j++) {
                segment->spine_ids[j] = segment->spine_ids[j + 1];
            }
            segment->num_spines--;
            break;
        }
    }

    // Mark spine as eliminated
    spine->state = SPINE_STATE_ELIMINATED;
    spine->stability = 0.0f;

    return true;
}

// ============================================================================
// Cable Theory and Electrical Properties
// ============================================================================

/**
 * WHAT: Calculate cable properties for segment
 * WHY:  Determine electrical behavior using cable theory
 * HOW:  Compute R_m, C_m, R_a from morphology
 */
static void calculate_cable_properties(dendritic_segment_t* segment) {
    // Guard: NULL segment
    if (!segment) {
        return;
    }

    // Surface area (cylinder)
    float surface_area = M_PI * segment->diameter * segment->length;  // μm²

    // Convert to cm²
    surface_area *= 1e-8f;  // 1 μm² = 1e-8 cm²

    // Membrane resistance (Ω)
    segment->R_m = DENDRITE_R_M / surface_area;

    // Membrane capacitance (μF)
    segment->C_m = DENDRITE_C_M * surface_area;

    // Axial resistance (Ω)
    float cross_section = M_PI * (segment->diameter / 2.0f) * (segment->diameter / 2.0f);  // μm²
    cross_section *= 1e-8f;  // Convert to cm²
    segment->R_a = (DENDRITE_R_A * segment->length * 1e-4f) / cross_section;  // Convert length to cm

    // Path resistance (accumulated)
    segment->path_resistance = segment->R_a;
}

/**
 * WHAT: Receive input at specific segment
 * WHY:  Inject synaptic current at compartment location
 * HOW:  Add current to segment voltage
 */
bool dendrite_receive_input(
    dendrite_t* dendrite,
    uint32_t segment_id,
    float current,
    uint64_t timestamp
) {
    // Guard: NULL dendrite
    if (!dendrite || segment_id >= dendrite->num_segments) {
        return false;
    }

    pthread_mutex_lock(&dendrite->lock);

    dendritic_segment_t* segment = &dendrite->segments[segment_id];

    // Add current (will be integrated in voltage update)
    segment->voltage += current * segment->R_m;

    // Update calcium based on NMDA component
    if (current > 0) {
        segment->calcium += current * 0.1f;  // Simplified calcium influx
    }

    // Update activity stats
    dendrite->activity.total_inputs++;
    dendrite->activity.mean_input_rate =
        (dendrite->activity.mean_input_rate * 0.99f) + 0.01f;

    pthread_mutex_unlock(&dendrite->lock);

    return true;
}

/**
 * WHAT: Compute total somatic current from dendrite
 * WHY:  Integrate dendritic inputs to soma
 * HOW:  Sum attenuated currents from all segments
 */
float dendrite_compute_somatic_current(dendrite_t* dendrite) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return 0.0f;
    }

    pthread_mutex_lock(&dendrite->lock);

    float total_current = 0.0f;

    // Sum attenuated voltages from all segments
    for (uint32_t i = 0; i < dendrite->num_segments; i++) {
        dendritic_segment_t* segment = &dendrite->segments[i];

        // Calculate attenuation factor
        float attenuation = calculate_attenuation(dendrite, i);

        // Add attenuated current
        total_current += segment->voltage * attenuation;
    }

    dendrite->somatic_voltage = total_current;

    pthread_mutex_unlock(&dendrite->lock);

    return total_current;
}

/**
 * WHAT: Calculate signal attenuation from segment to soma
 * WHY:  Determine how much signal reaches soma (cable theory)
 * HOW:  Use path resistance and length constant
 */
static float calculate_attenuation(dendrite_t* dendrite, uint32_t segment_id) {
    // Guard: Invalid inputs
    if (!dendrite || segment_id >= dendrite->num_segments) {
        return 0.0f;
    }

    dendritic_segment_t* segment = &dendrite->segments[segment_id];

    // Length constant λ = sqrt(R_m / R_a)
    float lambda = sqrtf(segment->R_m / segment->R_a);

    // Attenuation factor: exp(-distance / lambda)
    float attenuation = expf(-segment->path_distance / lambda);

    return attenuation;
}

/**
 * WHAT: Get attenuation factor for segment
 * WHY:  Query cable properties for analysis
 * HOW:  Calculate and return attenuation
 */
float dendrite_get_attenuation(dendrite_t* dendrite, uint32_t segment_id) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return 0.0f;
    }

    pthread_mutex_lock(&dendrite->lock);
    float attenuation = calculate_attenuation(dendrite, segment_id);
    pthread_mutex_unlock(&dendrite->lock);

    return attenuation;
}

// ============================================================================
// Time Evolution
// ============================================================================

/**
 * WHAT: Advance dendrite simulation by one timestep
 * WHY:  Update voltage, calcium, and plasticity
 * HOW:  Integrate differential equations for all segments
 */
void dendrite_step(dendrite_t* dendrite, float dt_ms, uint64_t timestamp) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return;
    }

    pthread_mutex_lock(&dendrite->lock);

    // Update all segments
    for (uint32_t i = 0; i < dendrite->num_segments; i++) {
        update_segment_voltage(&dendrite->segments[i], dt_ms);
        update_segment_calcium(&dendrite->segments[i], dt_ms);
    }

    // Update mean voltage
    float voltage_sum = 0.0f;
    for (uint32_t i = 0; i < dendrite->num_segments; i++) {
        voltage_sum += dendrite->segments[i].voltage;
    }
    dendrite->mean_voltage = voltage_sum / (float)dendrite->num_segments;

    // Update calcium level (mean across segments)
    float calcium_sum = 0.0f;
    for (uint32_t i = 0; i < dendrite->num_segments; i++) {
        calcium_sum += dendrite->segments[i].calcium;
    }
    dendrite->calcium_level = calcium_sum / (float)dendrite->num_segments;

    // Update activity stats
    dendrite->activity.mean_voltage = dendrite->mean_voltage;
    dendrite->activity.mean_calcium = dendrite->calcium_level;

    pthread_mutex_unlock(&dendrite->lock);
}

/**
 * WHAT: Update segment voltage using cable equation
 * WHY:  Model voltage dynamics in compartment
 * HOW:  dV/dt = (I_syn - V/R_m + I_axial) / C_m
 */
static void update_segment_voltage(dendritic_segment_t* segment, float dt_ms) {
    // Guard: NULL segment
    if (!segment) {
        return;
    }

    // Leak current
    float I_leak = -segment->voltage / segment->R_m;

    // Axial current (simplified - from parent)
    float I_axial = 0.0f;  // Would need parent reference for full implementation

    // Active currents (if segment has active properties)
    float I_active = 0.0f;
    if (segment->has_active_properties && segment->voltage > 0.5f) {
        // Simplified Hodgkin-Huxley
        I_active = -segment->g_na * segment->voltage;  // Sodium
        I_active += -segment->g_k * (segment->voltage + 0.012f);  // Potassium
    }

    // Total current
    float I_total = I_leak + I_axial + I_active;

    // Update voltage: dV/dt = I / C_m
    float dV = (I_total / segment->C_m) * (dt_ms / 1000.0f);  // Convert ms to s
    segment->voltage += dV;

    // Clamp voltage
    if (segment->voltage < -0.080f) segment->voltage = -0.080f;  // -80 mV
    if (segment->voltage > 0.040f) segment->voltage = 0.040f;    // +40 mV
}

/**
 * WHAT: Update segment calcium concentration
 * WHY:  Model calcium dynamics for plasticity
 * HOW:  Exponential decay with buffering
 */
static void update_segment_calcium(dendritic_segment_t* segment, float dt_ms) {
    // Guard: NULL segment
    if (!segment) {
        return;
    }

    // Calcium decay: dCa/dt = -Ca / tau
    float decay_factor = expf(-dt_ms / CALCIUM_DECAY_TAU);
    segment->calcium *= decay_factor;

    // Clamp calcium
    if (segment->calcium < 0.0f) segment->calcium = 0.0f;
    if (segment->calcium > CALCIUM_BUFFER_CAPACITY) {
        segment->calcium = CALCIUM_BUFFER_CAPACITY;
    }
}

/**
 * WHAT: Update calcium level for dendrite
 * WHY:  Integrate calcium across all segments
 * HOW:  Average segment calcium levels
 */
void dendrite_update_calcium(dendrite_t* dendrite, float dt_ms) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return;
    }

    pthread_mutex_lock(&dendrite->lock);

    // Already updated in dendrite_step
    // This function provides explicit calcium update interface

    pthread_mutex_unlock(&dendrite->lock);
}

// ============================================================================
// Plasticity
// ============================================================================

/**
 * WHAT: Induce LTP at dendrite
 * WHY:  Strengthen synaptic connections
 * HOW:  Increase spine receptor density based on calcium
 */
void dendrite_induce_ltp(dendrite_t* dendrite, uint32_t spine_id, float magnitude) {
    // Guard: Invalid inputs
    if (!dendrite || spine_id >= dendrite->num_spines) {
        return;
    }

    pthread_mutex_lock(&dendrite->lock);

    dendritic_spine_t* spine = &dendrite->spines[spine_id];

    // Check calcium threshold
    if (spine->calcium >= dendrite->ltp_threshold) {
        // Increase synaptic weight
        spine->synaptic_weight += magnitude * 0.1f;

        // Increase receptor density
        spine->ampa_receptors *= (1.0f + magnitude * 0.05f);
        spine->nmda_receptors *= (1.0f + magnitude * 0.03f);

        // Increase spine size (mushroom-like)
        spine->head_diameter *= (1.0f + magnitude * 0.02f);

        // Update stability
        spine->stability = fminf(1.0f, spine->stability + magnitude * 0.1f);

        // Update activity stats
        dendrite->activity.ltp_events++;

        nimcp_log(LOG_LEVEL_DEBUG, "dendrite_induce_ltp: LTP at spine %u (magnitude=%.3f)",
                  spine_id, magnitude);
    }

    pthread_mutex_unlock(&dendrite->lock);
}

/**
 * WHAT: Induce LTD at dendrite
 * WHY:  Weaken synaptic connections
 * HOW:  Decrease spine receptor density
 */
void dendrite_induce_ltd(dendrite_t* dendrite, uint32_t spine_id, float magnitude) {
    // Guard: Invalid inputs
    if (!dendrite || spine_id >= dendrite->num_spines) {
        return;
    }

    pthread_mutex_lock(&dendrite->lock);

    dendritic_spine_t* spine = &dendrite->spines[spine_id];

    // Check calcium threshold (lower than LTP)
    if (spine->calcium >= dendrite->ltd_threshold &&
        spine->calcium < dendrite->ltp_threshold) {
        // Decrease synaptic weight
        spine->synaptic_weight -= magnitude * 0.05f;
        if (spine->synaptic_weight < 0.1f) spine->synaptic_weight = 0.1f;

        // Decrease receptor density
        spine->ampa_receptors *= (1.0f - magnitude * 0.03f);
        spine->nmda_receptors *= (1.0f - magnitude * 0.02f);

        // Decrease spine size
        spine->head_diameter *= (1.0f - magnitude * 0.01f);

        // Update stability
        spine->stability = fmaxf(0.0f, spine->stability - magnitude * 0.1f);

        // Update activity stats
        dendrite->activity.ltd_events++;

        nimcp_log(LOG_LEVEL_DEBUG, "dendrite_induce_ltd: LTD at spine %u (magnitude=%.3f)",
                  spine_id, magnitude);
    }

    pthread_mutex_unlock(&dendrite->lock);
}

/**
 * WHAT: Update structural plasticity
 * WHY:  Add/remove spines based on activity
 * HOW:  Check stability, prune weak spines
 */
void dendrite_update_structural_plasticity(dendrite_t* dendrite, uint64_t timestamp) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return;
    }

    // Guard: Structural plasticity disabled
    if (dendrite->structural_plasticity <= 0.0f) {
        return;
    }

    pthread_mutex_lock(&dendrite->lock);

    // Check each spine for elimination
    for (uint32_t i = 0; i < dendrite->num_spines; i++) {
        dendritic_spine_t* spine = &dendrite->spines[i];

        // Skip already eliminated spines
        if (spine->state == SPINE_STATE_ELIMINATED) {
            continue;
        }

        // Prune unstable spines
        if (spine->stability < 0.2f) {
            spine->state = SPINE_STATE_PRUNING;

            // Gradual elimination
            spine->stability -= dendrite->structural_plasticity;
            if (spine->stability <= 0.0f) {
                dendrite_remove_spine(dendrite, i);
                dendrite->activity.spine_eliminations++;
            }
        }
    }

    pthread_mutex_unlock(&dendrite->lock);
}

// ============================================================================
// Query Functions
// ============================================================================

/**
 * WHAT: Check if dendrite is in plateau potential state
 * WHY:  Detect dendritic spikes and nonlinear integration
 * HOW:  Check if voltage exceeds threshold
 */
bool dendrite_is_in_plateau(dendrite_t* dendrite) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return false;
    }

    pthread_mutex_lock(&dendrite->lock);

    // Plateau threshold: ~40 mV above rest
    bool in_plateau = (dendrite->mean_voltage > 0.030f);

    pthread_mutex_unlock(&dendrite->lock);

    return in_plateau;
}

/**
 * WHAT: Get spine by synapse ID
 * WHY:  Look up spine for synapse
 * HOW:  Linear search through spines
 */
dendritic_spine_t* dendrite_get_spine_by_synapse(
    dendrite_t* dendrite,
    uint32_t synapse_id
) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return NULL;
    }

    pthread_mutex_lock(&dendrite->lock);

    dendritic_spine_t* result = NULL;

    for (uint32_t i = 0; i < dendrite->num_spines; i++) {
        if (dendrite->spines[i].synapse_id == synapse_id) {
            result = &dendrite->spines[i];
            break;
        }
    }

    pthread_mutex_unlock(&dendrite->lock);

    return result;
}

/**
 * WHAT: Calculate total surface area
 * WHY:  Determine dendrite size for normalization
 * HOW:  Sum cylinder surface areas of all segments
 */
float dendrite_calculate_surface_area(dendrite_t* dendrite) {
    // Guard: NULL dendrite
    if (!dendrite) {
        return 0.0f;
    }

    pthread_mutex_lock(&dendrite->lock);

    float total_area = 0.0f;

    for (uint32_t i = 0; i < dendrite->num_segments; i++) {
        dendritic_segment_t* seg = &dendrite->segments[i];
        total_area += M_PI * seg->diameter * seg->length;
    }

    dendrite->surface_area = total_area;

    pthread_mutex_unlock(&dendrite->lock);

    return total_area;
}

/**
 * WHAT: Get dendrite activity statistics
 * WHY:  Monitor dendrite function
 * HOW:  Return activity struct
 */
dendrite_activity_stats_t dendrite_get_activity_stats(dendrite_t* dendrite) {
    dendrite_activity_stats_t stats = {0};

    // Guard: NULL dendrite
    if (!dendrite) {
        return stats;
    }

    pthread_mutex_lock(&dendrite->lock);
    stats = dendrite->activity;
    pthread_mutex_unlock(&dendrite->lock);

    return stats;
}

// ============================================================================
// Dendrite Network Management
// ============================================================================

/**
 * WHAT: Create dendrite network manager
 * WHY:  Manage collection of dendrites
 * HOW:  Allocate network structure
 */
dendrite_network_t* dendrite_network_create(uint32_t max_dendrites) {
    // Guard: Invalid size
    if (max_dendrites == 0) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_network_create: Invalid max_dendrites");
        return NULL;
    }

    // Allocate network
    dendrite_network_t* network = (dendrite_network_t*)nimcp_calloc(
        1, sizeof(dendrite_network_t)
    );
    if (!network) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_network_create: Failed to allocate network");
        return NULL;
    }

    // Allocate dendrite array
    network->dendrites = (dendrite_t**)nimcp_calloc(
        max_dendrites, sizeof(dendrite_t*)
    );
    if (!network->dendrites) {
        nimcp_free(network);
        return NULL;
    }

    network->num_dendrites = 0;
    network->max_dendrites = max_dendrites;
    pthread_mutex_init(&network->lock, NULL);

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_network_create: Created network (max=%u)",
              max_dendrites);

    return network;
}

/**
 * WHAT: Destroy dendrite network
 * WHY:  Clean up all dendrites
 * HOW:  Destroy each dendrite, free arrays
 */
void dendrite_network_destroy(dendrite_network_t* network) {
    // Guard: NULL network
    if (!network) {
        return;
    }

    // Destroy all dendrites
    for (uint32_t i = 0; i < network->num_dendrites; i++) {
        if (network->dendrites[i]) {
            dendrite_destroy(network->dendrites[i]);
        }
    }

    // Free arrays
    nimcp_free(network->dendrites);

    // Destroy mutex
    pthread_mutex_destroy(&network->lock);

    // Free network
    nimcp_free(network);
}

/**
 * WHAT: Add dendrite to network
 * WHY:  Register dendrite for network-wide operations
 * HOW:  Add to array if space available, thread-safe
 */
bool dendrite_network_add(dendrite_network_t* network, dendrite_t* dendrite) {
    // Guard: NULL parameters
    if (!network || !dendrite) {
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_network_add: NULL parameter");
        return false;
    }

    pthread_mutex_lock(&network->lock);

    // Guard: Network full
    if (network->num_dendrites >= network->max_dendrites) {
        pthread_mutex_unlock(&network->lock);
        nimcp_log(LOG_LEVEL_ERROR, "dendrite_network_add: Network full (%u/%u)",
                  network->num_dendrites, network->max_dendrites);
        return false;
    }

    // Add dendrite
    network->dendrites[network->num_dendrites] = dendrite;
    network->num_dendrites++;

    pthread_mutex_unlock(&network->lock);

    nimcp_log(LOG_LEVEL_DEBUG, "dendrite_network_add: Added dendrite %u (total=%u)",
              dendrite->id, network->num_dendrites);

    return true;
}

/**
 * WHAT: Advance all dendrites in network
 * WHY:  Update entire dendritic system
 * HOW:  Call dendrite_step for each dendrite
 */
void dendrite_network_step(
    dendrite_network_t* network,
    float dt_ms,
    uint64_t timestamp
) {
    // Guard: NULL network
    if (!network) {
        return;
    }

    pthread_mutex_lock(&network->lock);

    for (uint32_t i = 0; i < network->num_dendrites; i++) {
        if (network->dendrites[i]) {
            dendrite_step(network->dendrites[i], dt_ms, timestamp);
        }
    }

    pthread_mutex_unlock(&network->lock);
}

/**
 * WHAT: Get network-wide statistics
 * WHY:  Monitor overall dendritic system
 * HOW:  Aggregate stats from all dendrites
 */
dendrite_network_stats_t dendrite_network_get_stats(dendrite_network_t* network) {
    dendrite_network_stats_t stats = {0};

    // Guard: NULL network
    if (!network) {
        return stats;
    }

    pthread_mutex_lock(&network->lock);

    stats.total_dendrites = network->num_dendrites;

    for (uint32_t i = 0; i < network->num_dendrites; i++) {
        dendrite_t* d = network->dendrites[i];
        if (!d) continue;

        stats.total_segments += d->num_segments;
        stats.total_spines += d->num_spines;
        stats.mean_voltage += d->mean_voltage;
        stats.mean_calcium += d->calcium_level;
        stats.total_ltp_events += d->activity.ltp_events;
        stats.total_ltd_events += d->activity.ltd_events;
        stats.total_spine_formations += d->activity.spine_formations;
        stats.total_spine_eliminations += d->activity.spine_eliminations;
    }

    // Calculate means
    if (network->num_dendrites > 0) {
        stats.mean_voltage /= (float)network->num_dendrites;
        stats.mean_calcium /= (float)network->num_dendrites;
    }

    pthread_mutex_unlock(&network->lock);

    return stats;
}
