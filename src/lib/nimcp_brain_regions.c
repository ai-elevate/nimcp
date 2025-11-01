/**
 * @file nimcp_brain_regions.c
 * @brief Implementation of modular brain architecture
 *
 * Provides hierarchical organization of neurons into:
 * - Brain regions (V1, A1, M1, etc.)
 * - Cortical layers (1-6)
 * - Minicolumns (vertical processing units)
 * - Inter-region connectivity
 */

#include "nimcp_brain_regions.h"
#include "nimcp_neuron_types.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_validate.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Generate unique region ID
 */
static uint32_t generate_region_id(void) {
    static uint32_t next_id = 1;
    return next_id++;
}

/**
 * @brief Get default layer proportions for region type
 *
 * Based on actual cortical layer thickness measurements
 */
static void get_layer_proportions(brain_region_type_t type, float proportions[LAYER_COUNT]) {
    // Default cortical proportions (approximate)
    switch (type) {
        case REGION_VISUAL_V1:
            // V1 has prominent Layer 4 (input from LGN)
            proportions[LAYER_1] = 0.05f;  // Thin
            proportions[LAYER_2] = 0.10f;
            proportions[LAYER_3] = 0.15f;
            proportions[LAYER_4] = 0.35f;  // Thick in V1
            proportions[LAYER_5] = 0.20f;
            proportions[LAYER_6] = 0.15f;
            break;

        case REGION_AUDITORY_A1:
            // A1 also has prominent Layer 4
            proportions[LAYER_1] = 0.05f;
            proportions[LAYER_2] = 0.12f;
            proportions[LAYER_3] = 0.18f;
            proportions[LAYER_4] = 0.30f;
            proportions[LAYER_5] = 0.20f;
            proportions[LAYER_6] = 0.15f;
            break;

        case REGION_MOTOR_M1:
            // M1 has prominent Layer 5 (output to spinal cord)
            proportions[LAYER_1] = 0.05f;
            proportions[LAYER_2] = 0.10f;
            proportions[LAYER_3] = 0.15f;
            proportions[LAYER_4] = 0.15f;
            proportions[LAYER_5] = 0.35f;  // Thick in M1
            proportions[LAYER_6] = 0.20f;
            break;

        case REGION_PREFRONTAL:
            // Prefrontal has large Layer 3 (associative)
            proportions[LAYER_1] = 0.05f;
            proportions[LAYER_2] = 0.15f;
            proportions[LAYER_3] = 0.30f;  // Thick
            proportions[LAYER_4] = 0.15f;
            proportions[LAYER_5] = 0.20f;
            proportions[LAYER_6] = 0.15f;
            break;

        default:
            // Standard cortical proportions
            proportions[LAYER_1] = 0.05f;
            proportions[LAYER_2] = 0.15f;
            proportions[LAYER_3] = 0.20f;
            proportions[LAYER_4] = 0.20f;
            proportions[LAYER_5] = 0.25f;
            proportions[LAYER_6] = 0.15f;
            break;
    }
}

/**
 * @brief Get appropriate neuron type for layer
 */
static neuron_type_extended_t get_layer_neuron_type(brain_region_type_t region_type,
                                                      cortical_layer_t layer) {
    switch (region_type) {
        case REGION_VISUAL_V1:
        case REGION_VISUAL_V2:
        case REGION_VISUAL_V4:
            // Visual cortex uses specialized visual neurons
            if (layer == LAYER_4) {
                return NEURON_VISUAL_EDGE; // Input layer: edge detectors
            } else if (layer == LAYER_2 || layer == LAYER_3) {
                return NEURON_VISUAL_ORIENTATION; // Orientation-selective
            } else if (layer == LAYER_5 || layer == LAYER_6) {
                return NEURON_PYRAMIDAL_L5_THICK; // Output layers
            }
            break;

        case REGION_VISUAL_MT:
            return NEURON_VISUAL_DIRECTION; // Motion-selective

        case REGION_AUDITORY_A1:
        case REGION_AUDITORY_A2:
            // Auditory cortex uses frequency-tuned neurons
            if (layer == LAYER_4) {
                return NEURON_AUDITORY_FREQUENCY; // Frequency-tuned input
            } else if (layer == LAYER_2 || layer == LAYER_3) {
                return NEURON_AUDITORY_ONSET; // Temporal features
            }
            break;

        case REGION_MOTOR_M1:
            if (layer == LAYER_5) {
                return NEURON_MOTOR_ALPHA; // Motoneurons
            } else if (layer == LAYER_2 || layer == LAYER_3) {
                return NEURON_MOTOR_PATTERN_GEN; // Pattern generators
            }
            break;

        default:
            break;
    }

    // Default: pyramidal neurons
    if (layer == LAYER_2 || layer == LAYER_3) {
        return NEURON_PYRAMIDAL_L23;
    } else if (layer == LAYER_5) {
        return NEURON_PYRAMIDAL_L5_THICK;
    } else if (layer == LAYER_6) {
        return NEURON_PYRAMIDAL_L6;
    }

    return NEURON_EXCITATORY; // Fallback
}

// ============================================================================
// BRAIN MODULE MANAGEMENT
// ============================================================================

brain_module_t* brain_module_create(uint32_t max_regions) {
    brain_module_t* brain = (brain_module_t*)nimcp_calloc(1, sizeof(brain_module_t));
    if (!brain) return NULL;

    brain->id = 1; // Fixed ID for now
    brain->max_regions = max_regions;

    brain->regions = (brain_region_t**)nimcp_calloc(max_regions, sizeof(brain_region_t*));
    if (!brain->regions) {
        nimcp_free(brain);
        return NULL;
    }

    brain->connections = NULL; // Allocated on demand
    brain->num_connections = 0;

    brain->enable_plasticity = true;
    brain->enable_glial = true;

    nimcp_mutex_init(&brain->lock, NULL);

    return brain;
}

void brain_module_destroy(brain_module_t* brain) {
    if (!brain) return;

    // Destroy all regions
    for (uint32_t i = 0; i < brain->num_regions; i++) {
        brain_region_destroy(brain->regions[i]);
    }

    nimcp_free(brain->regions);

    // Destroy connections
    if (brain->connections) {
        for (uint32_t i = 0; i < brain->num_connections; i++) {
            nimcp_free(brain->connections[i]);
        }
        nimcp_free(brain->connections);
    }

    nimcp_mutex_destroy(&brain->lock);
    nimcp_free(brain);
}

nimcp_result_t brain_module_add_region(brain_module_t* brain, brain_region_t* region) {
    if (!brain || !region) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (brain->num_regions >= brain->max_regions) {
        return NIMCP_ERROR; // Full
    }

    nimcp_mutex_lock(&brain->lock);

    brain->regions[brain->num_regions] = region;
    brain->num_regions++;
    brain->total_neurons += region->total_neurons;

    nimcp_mutex_unlock(&brain->lock);

    return NIMCP_SUCCESS;
}

brain_region_t* brain_module_get_region(brain_module_t* brain, uint32_t region_id) {
    if (!brain) return NULL;

    for (uint32_t i = 0; i < brain->num_regions; i++) {
        if (brain->regions[i]->id == region_id) {
            return brain->regions[i];
        }
    }

    return NULL;
}

brain_region_t* brain_module_get_region_by_type(brain_module_t* brain,
                                                 brain_region_type_t type) {
    if (!brain) return NULL;

    for (uint32_t i = 0; i < brain->num_regions; i++) {
        if (brain->regions[i]->type == type) {
            return brain->regions[i];
        }
    }

    return NULL;
}

// ============================================================================
// BRAIN REGION MANAGEMENT
// ============================================================================

brain_region_t* brain_region_create(brain_region_type_t type, uint32_t num_neurons) {
    brain_region_t* region = (brain_region_t*)nimcp_calloc(1, sizeof(brain_region_t));
    if (!region) return NULL;

    region->id = generate_region_id();
    region->type = type;
    region->total_neurons = num_neurons;

    // Set human-readable name
    const char* name = brain_region_get_name(type);
    strncpy(region->name, name, sizeof(region->name) - 1);

    // Create underlying neural network using public API
    network_config_t config = {0};
    config.num_neurons = num_neurons;
    config.ei_ratio = 0.8f;  // 80% excitatory
    config.learning_rate = 0.01f;
    config.target_activity = 0.1f;
    config.min_weight = -1.0f;
    config.max_weight = 1.0f;

    region->network = neural_network_create(&config);
    if (!region->network) {
        nimcp_free(region);
        return NULL;
    }

    // Organize into layers with biologically realistic proportions
    float proportions[LAYER_COUNT];
    get_layer_proportions(type, proportions);

    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        region->layer_sizes[layer] = (uint32_t)(num_neurons * proportions[layer]);
    }

    // Adjust for rounding errors - give extra neurons to Layer 4 (input layer)
    uint32_t allocated = 0;
    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        allocated += region->layer_sizes[layer];
    }
    if (allocated < num_neurons) {
        region->layer_sizes[LAYER_4] += (num_neurons - allocated);
    }

    // Allocate extended neuron type arrays
    region->neuron_extended_types = (neuron_type_extended_t*)nimcp_calloc(num_neurons, sizeof(neuron_type_extended_t));
    region->neuron_type_params = (neuron_type_params_t*)nimcp_calloc(num_neurons, sizeof(neuron_type_params_t));
    if (!region->neuron_extended_types || !region->neuron_type_params) {
        nimcp_free(region->neuron_extended_types);
        nimcp_free(region->neuron_type_params);
        neural_network_destroy(region->network);
        nimcp_free(region);
        return NULL;
    }

    // Assign neuron types based on region and layer
    uint32_t neuron_idx = 0;
    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        for (uint32_t i = 0; i < region->layer_sizes[layer]; i++) {
            if (neuron_idx < num_neurons) {
                // Get specialized type for this layer
                neuron_type_extended_t ext_type = get_layer_neuron_type(type, (cortical_layer_t)layer);
                region->neuron_extended_types[neuron_idx] = ext_type;

                // Get default parameters for this type
                neuron_type_get_default_params(ext_type, &region->neuron_type_params[neuron_idx]);

                neuron_idx++;
            }
        }
    }

    region->minicolumns = NULL;
    region->num_minicolumns = 0;

    region->activity_level = 0.0f;
    region->last_update = 0;

    nimcp_mutex_init(&region->lock, NULL);

    return region;
}

void brain_region_destroy(brain_region_t* region) {
    if (!region) return;

    // Destroy minicolumns
    if (region->minicolumns) {
        for (uint32_t i = 0; i < region->num_minicolumns; i++) {
            if (region->minicolumns[i]) {
                for (int layer = 0; layer < LAYER_COUNT; layer++) {
                    nimcp_free(region->minicolumns[i]->layer_neuron_ids[layer]);
                }
                nimcp_free(region->minicolumns[i]);
            }
        }
        nimcp_free(region->minicolumns);
    }

    // Free extended neuron types
    nimcp_free(region->neuron_extended_types);
    nimcp_free(region->neuron_type_params);

    // Destroy glial integration
    if (region->glial) {
        glial_integration_destroy(region->glial);
    }

    // Destroy neural network
    if (region->network) {
        neural_network_destroy(region->network);
    }

    nimcp_free(region->input_regions);
    nimcp_free(region->output_regions);

    nimcp_mutex_destroy(&region->lock);
    nimcp_free(region);
}

nimcp_result_t brain_region_organize_columns(brain_region_t* region,
                                               uint32_t columns_x,
                                               uint32_t columns_y) {
    if (!region || columns_x == 0 || columns_y == 0) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    uint32_t num_columns = columns_x * columns_y;

    region->minicolumns = (brain_minicolumn_t**)nimcp_calloc(num_columns, sizeof(brain_minicolumn_t*));
    if (!region->minicolumns) {
        return NIMCP_ERROR;
    }

    region->num_minicolumns = num_columns;
    region->minicolumns_x = columns_x;
    region->minicolumns_y = columns_y;

    uint32_t neurons_per_column = region->total_neurons / num_columns;

    // Create each minicolumn
    for (uint32_t col_idx = 0; col_idx < num_columns; col_idx++) {
        brain_minicolumn_t* col = (brain_minicolumn_t*)nimcp_calloc(1, sizeof(brain_minicolumn_t));
        if (!col) continue;

        col->id = col_idx;
        col->region_id = region->id;

        // Position in 2D grid
        col->x = (float)(col_idx % columns_x) / (float)columns_x;
        col->y = (float)(col_idx / columns_x) / (float)columns_y;

        // Distribute neurons across layers proportionally
        uint32_t neuron_offset = col_idx * neurons_per_column;

        for (int layer = 0; layer < LAYER_COUNT; layer++) {
            uint32_t layer_neurons_in_col = region->layer_sizes[layer] / num_columns;
            if (layer_neurons_in_col > 0) {
                col->layer_neuron_ids[layer] = (uint32_t*)nimcp_calloc(layer_neurons_in_col, sizeof(uint32_t));
                col->layer_neuron_counts[layer] = layer_neurons_in_col;

                // Assign neuron IDs
                for (uint32_t n = 0; n < layer_neurons_in_col; n++) {
                    col->layer_neuron_ids[layer][n] = neuron_offset + n;
                }
                neuron_offset += layer_neurons_in_col;
            } else {
                col->layer_neuron_ids[layer] = NULL;
                col->layer_neuron_counts[layer] = 0;
            }
        }

        region->minicolumns[col_idx] = col;
    }

    return NIMCP_SUCCESS;
}

uint32_t brain_region_get_layer_neurons(brain_region_t* region,
                                         cortical_layer_t layer,
                                         uint32_t* out_neuron_ids,
                                         uint32_t max_neurons) {
    if (!region || !out_neuron_ids || layer >= LAYER_COUNT) {
        return 0;
    }

    uint32_t layer_size = region->layer_sizes[layer];
    uint32_t count = (layer_size < max_neurons) ? layer_size : max_neurons;

    // Calculate starting neuron ID for this layer
    uint32_t start_id = 0;
    for (int l = 0; l < layer; l++) {
        start_id += region->layer_sizes[l];
    }

    for (uint32_t i = 0; i < count; i++) {
        out_neuron_ids[i] = start_id + i;
    }

    return count;
}

// ============================================================================
// INTER-REGION CONNECTIVITY
// ============================================================================

nimcp_result_t brain_module_connect_regions(brain_module_t* brain,
                                             uint32_t source_region_id,
                                             uint32_t target_region_id,
                                             float connection_density) {
    if (!brain) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    brain_region_t* source = brain_module_get_region(brain, source_region_id);
    brain_region_t* target = brain_module_get_region(brain, target_region_id);

    if (!source || !target) {
        return NIMCP_ERROR; // Region not found
    }

    // Create connection record
    brain_connection_t* conn = (brain_connection_t*)nimcp_calloc(1, sizeof(brain_connection_t));
    if (!conn) {
        return NIMCP_ERROR;
    }

    conn->source_region_id = source_region_id;
    conn->target_region_id = target_region_id;
    conn->connection_strength = connection_density;
    conn->feedforward = true;

    // Default: Layer 2/3 → Layer 4 (feedforward)
    conn->source_layer = LAYER_3;
    conn->target_layer = LAYER_4;

    // Add to brain's connection list
    brain->connections = (brain_connection_t**)nimcp_realloc(
        brain->connections,
        (brain->num_connections + 1) * sizeof(brain_connection_t*));

    brain->connections[brain->num_connections] = conn;
    brain->num_connections++;

    return NIMCP_SUCCESS;
}

nimcp_result_t brain_module_connect_layers(brain_module_t* brain,
                                            uint32_t source_region_id,
                                            cortical_layer_t source_layer,
                                            uint32_t target_region_id,
                                            cortical_layer_t target_layer,
                                            float connection_density) {
    if (!brain) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    brain_connection_t* conn = (brain_connection_t*)nimcp_calloc(1, sizeof(brain_connection_t));
    if (!conn) {
        return NIMCP_ERROR;
    }

    conn->source_region_id = source_region_id;
    conn->source_layer = source_layer;
    conn->target_region_id = target_region_id;
    conn->target_layer = target_layer;
    conn->connection_strength = connection_density;
    conn->feedforward = (target_layer == LAYER_4);

    brain->connections = (brain_connection_t**)nimcp_realloc(
        brain->connections,
        (brain->num_connections + 1) * sizeof(brain_connection_t*));

    brain->connections[brain->num_connections] = conn;
    brain->num_connections++;

    return NIMCP_SUCCESS;
}

// ============================================================================
// SENSORY INPUT & PROCESSING
// ============================================================================

nimcp_result_t brain_region_process_input(brain_region_t* region,
                                           const float* input,
                                           uint32_t input_size,
                                           uint64_t timestamp) {
    if (!region || !input) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Distribute input to Layer 4 neurons (thalamic input layer)
    uint32_t layer4_size = region->layer_sizes[LAYER_4];
    if (layer4_size == 0) {
        return NIMCP_ERROR; // No input layer
    }

    // Calculate starting index for Layer 4
    uint32_t layer4_start = 0;
    for (int l = 0; l < LAYER_4; l++) {
        layer4_start += region->layer_sizes[l];
    }

    // Distribute input across Layer 4 neurons using public API
    for (uint32_t i = 0; i < layer4_size && i < input_size; i++) {
        uint32_t neuron_idx = layer4_start + i;
        if (neuron_idx < region->total_neurons) {
            float current_state = 0.0f;
            neural_network_get_neuron_state(region->network, neuron_idx, &current_state);
            float new_state = current_state + input[i % input_size];
            neural_network_update_neuron(region->network, neuron_idx, new_state, timestamp);
        }
    }

    region->last_update = timestamp;

    return NIMCP_SUCCESS;
}

uint32_t brain_region_get_output(brain_region_t* region,
                                  float* output,
                                  uint32_t output_size) {
    if (!region || !output) {
        return 0;
    }

    // Collect activity from Layer 5 neurons (output layer)
    uint32_t layer5_size = region->layer_sizes[LAYER_5];
    if (layer5_size == 0) {
        return 0;
    }

    // Calculate starting index for Layer 5
    uint32_t layer5_start = 0;
    for (int l = 0; l < LAYER_5; l++) {
        layer5_start += region->layer_sizes[l];
    }

    uint32_t count = (layer5_size < output_size) ? layer5_size : output_size;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t neuron_idx = layer5_start + i;
        if (neuron_idx < region->total_neurons) {
            neural_network_get_neuron_state(region->network, neuron_idx, &output[i]);
        }
    }

    return count;
}

// ============================================================================
// SIMULATION & STEPPING
// ============================================================================

nimcp_result_t brain_module_step(brain_module_t* brain, uint64_t delta_t) {
    if (!brain) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    brain->current_time += delta_t;

    // Step all regions
    for (uint32_t i = 0; i < brain->num_regions; i++) {
        brain_region_step(brain->regions[i], delta_t);
    }

    // TODO: Propagate signals between regions via connections

    return NIMCP_SUCCESS;
}

nimcp_result_t brain_region_step(brain_region_t* region, uint64_t delta_t) {
    if (!region) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Simple time update for now
    region->last_update += delta_t;

    // TODO: Step underlying neural network
    // TODO: Update glial cells

    return NIMCP_SUCCESS;
}

// ============================================================================
// STATISTICS & MONITORING
// ============================================================================

nimcp_result_t brain_region_get_stats(brain_region_t* region,
                                       brain_region_stats_t* stats) {
    if (!region || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(stats, 0, sizeof(brain_region_stats_t));

    stats->total_neurons = region->total_neurons;
    stats->num_minicolumns = region->num_minicolumns;

    // Calculate activity per layer
    for (int layer = 0; layer < LAYER_COUNT; layer++) {
        uint32_t layer_start = 0;
        for (int l = 0; l < layer; l++) {
            layer_start += region->layer_sizes[l];
        }

        float layer_activity = 0.0f;
        uint32_t active_count = 0;
        for (uint32_t i = 0; i < region->layer_sizes[layer]; i++) {
            uint32_t neuron_idx = layer_start + i;
            if (neuron_idx < region->total_neurons) {
                float state = 0.0f;
                neural_network_get_neuron_state(region->network, neuron_idx, &state);
                // Normalize: treat state > threshold (1.0) as active
                // Use sigmoid-like normalization: tanh(state) to map to [0,1]
                float normalized = tanhf(fabsf(state));
                layer_activity += normalized;
                if (fabsf(state) > 0.5f) {
                    active_count++;
                }
            }
        }

        if (region->layer_sizes[layer] > 0) {
            stats->layer_activity[layer] = layer_activity / region->layer_sizes[layer];
        }
    }

    // Overall average
    float total_activity = 0.0f;
    for (int l = 0; l < LAYER_COUNT; l++) {
        total_activity += stats->layer_activity[l];
    }
    stats->avg_activity = total_activity / LAYER_COUNT;

    return NIMCP_SUCCESS;
}

const char* brain_region_get_name(brain_region_type_t type) {
    switch (type) {
        case REGION_VISUAL_V1:       return "Primary Visual Cortex (V1)";
        case REGION_VISUAL_V2:       return "Secondary Visual Cortex (V2)";
        case REGION_VISUAL_V4:       return "Visual Area V4";
        case REGION_VISUAL_MT:       return "Middle Temporal (MT/V5)";
        case REGION_VISUAL_IT:       return "Inferior Temporal (IT)";

        case REGION_AUDITORY_A1:     return "Primary Auditory Cortex (A1)";
        case REGION_AUDITORY_A2:     return "Secondary Auditory Cortex (A2)";
        case REGION_AUDITORY_BELT:   return "Auditory Belt Region";
        case REGION_AUDITORY_PARABELT: return "Auditory Parabelt";

        case REGION_MOTOR_M1:        return "Primary Motor Cortex (M1)";
        case REGION_MOTOR_PREMOTOR:  return "Premotor Cortex";
        case REGION_MOTOR_SMA:       return "Supplementary Motor Area (SMA)";

        case REGION_SOMATOSENSORY_S1: return "Primary Somatosensory Cortex (S1)";
        case REGION_SOMATOSENSORY_S2: return "Secondary Somatosensory Cortex (S2)";

        case REGION_PREFRONTAL:      return "Prefrontal Cortex (PFC)";
        case REGION_PARIETAL:        return "Parietal Cortex";
        case REGION_TEMPORAL:        return "Temporal Cortex";

        case REGION_THALAMUS:        return "Thalamus";
        case REGION_HIPPOCAMPUS:     return "Hippocampus";
        case REGION_BASAL_GANGLIA:   return "Basal Ganglia";
        case REGION_CEREBELLUM:      return "Cerebellum";

        default:                     return "Unknown Region";
    }
}
