/**
 * @file nimcp_glial_integration.c
 * @brief Implementation of glial-neuron integration
 *
 * TDD STATUS: Stub implementation for RED phase
 * - All functions present but minimal implementation
 * - Basic memory management working
 * - Will implement full functionality in GREEN phase
 */

#include "nimcp_glial_integration.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Dynamic array for storing lists of IDs in reverse mappings
 */
typedef struct {
    uint32_t* ids;      // Array of IDs
    uint32_t count;     // Number of IDs
    uint32_t capacity;  // Allocated capacity
} id_list_t;

static id_list_t* id_list_create(void) {
    id_list_t* list = (id_list_t*)nimcp_malloc(sizeof(id_list_t));
    if (!list) return NULL;

    list->capacity = 8;  // Initial capacity
    list->count = 0;
    list->ids = (uint32_t*)nimcp_malloc(sizeof(uint32_t) * list->capacity);
    if (!list->ids) {
        nimcp_free(list);
        return NULL;
    }

    return list;
}

static void id_list_destroy(id_list_t* list) {
    if (!list) return;
    if (list->ids) nimcp_free(list->ids);
    nimcp_free(list);
}

// Value destructor for hash tables storing id_list_t*
static void id_list_value_destructor(void* value, size_t value_size) {
    (void)value_size;  // Unused parameter
    if (value) {
        id_list_t* list = *(id_list_t**)value;
        id_list_destroy(list);
    }
}


static bool id_list_add(id_list_t* list, uint32_t id) {
    if (!list) return false;

    // Check if already exists
    for (uint32_t i = 0; i < list->count; i++) {
        if (list->ids[i] == id) {
            return true; // Already exists, not an error
        }
    }

    // Expand if needed
    if (list->count >= list->capacity) {
        uint32_t new_capacity = list->capacity * 2;
        uint32_t* new_ids = (uint32_t*)nimcp_realloc(list->ids, sizeof(uint32_t) * new_capacity);
        if (!new_ids) return false;

        list->ids = new_ids;
        list->capacity = new_capacity;
    }

    list->ids[list->count++] = id;
    return true;
}

/**
 * @brief Generate synapse ID from pre/post neuron IDs
 */
static inline uint32_t make_synapse_id(uint32_t pre_neuron_id, uint32_t post_neuron_id) {
    return pre_neuron_id * 10000 + post_neuron_id;
}

// ============================================================================
// CREATION & DESTRUCTION
// ============================================================================

glial_integration_t* glial_integration_create(neural_network_t network, uint32_t max_mappings) {
    // Integration tests expect NULL network to be rejected
    if (!network || max_mappings == 0) {
        return NULL;
    }

    glial_integration_t* gi = (glial_integration_t*)nimcp_malloc(sizeof(glial_integration_t));
    if (!gi) {
        return NULL;
    }

    memset(gi, 0, sizeof(glial_integration_t));

    gi->network = network;

    // Config for forward mappings (glial_id → uint32_t)
    hash_table_config_t forward_config = {
        .initial_buckets = max_mappings > 256 ? max_mappings : 256,
        .key_type = HASH_KEY_UINT32,
        .hash_algorithm = HASH_ALG_MURMUR3,
        .custom_hash_fn = NULL,
        .custom_compare_fn = NULL,
        .value_destructor = NULL,  // Values are uint32_t, no cleanup needed
        .case_insensitive = false,
        .thread_safe = false
    };

    // Config for reverse mappings (glial_id → id_list_t*)
    hash_table_config_t reverse_config = {
        .initial_buckets = max_mappings > 256 ? max_mappings : 256,
        .key_type = HASH_KEY_UINT32,
        .hash_algorithm = HASH_ALG_MURMUR3,
        .custom_hash_fn = NULL,
        .custom_compare_fn = NULL,
        .value_destructor = id_list_value_destructor,  // Auto-cleanup
        .case_insensitive = false,
        .thread_safe = false
    };

    // Forward mappings
    gi->synapse_to_astrocyte = hash_table_create(&forward_config);
    gi->neuron_to_oligodendrocyte = hash_table_create(&forward_config);
    gi->synapse_to_microglia = hash_table_create(&forward_config);

    // Reverse mappings
    gi->astrocyte_to_synapses = hash_table_create(&reverse_config);
    gi->oligodendrocyte_to_neurons = hash_table_create(&reverse_config);
    gi->microglia_to_synapses = hash_table_create(&reverse_config);

    if (!gi->synapse_to_astrocyte || !gi->neuron_to_oligodendrocyte ||
        !gi->synapse_to_microglia || !gi->astrocyte_to_synapses ||
        !gi->oligodendrocyte_to_neurons || !gi->microglia_to_synapses) {
        glial_integration_destroy(gi);
        return NULL;
    }

    // Initialize statistics
    gi->total_astrocyte_modulations = 0;
    gi->total_oligodendrocyte_myelinations = 0;
    gi->total_microglia_prunings = 0;

    // Default: all features disabled
    gi->enable_astrocyte_modulation = false;
    gi->enable_oligodendrocyte_myelination = false;
    gi->enable_microglia_pruning = false;

    // Initialize lock
    nimcp_mutex_init(&gi->lock, NULL);

    return gi;
}

void glial_integration_destroy(glial_integration_t* gi) {
    if (!gi) return;

    if (gi->synapse_to_astrocyte) {
        hash_table_destroy(gi->synapse_to_astrocyte);
    }
    if (gi->neuron_to_oligodendrocyte) {
        hash_table_destroy(gi->neuron_to_oligodendrocyte);
    }
    if (gi->synapse_to_microglia) {
        hash_table_destroy(gi->synapse_to_microglia);
    }
    if (gi->astrocyte_to_synapses) {
        hash_table_destroy(gi->astrocyte_to_synapses);
    }
    if (gi->oligodendrocyte_to_neurons) {
        hash_table_destroy(gi->oligodendrocyte_to_neurons);
    }
    if (gi->microglia_to_synapses) {
        hash_table_destroy(gi->microglia_to_synapses);
    }

    nimcp_mutex_destroy(&gi->lock);
    nimcp_free(gi);
}

// ============================================================================
// GLIAL NETWORK ASSIGNMENT
// ============================================================================

nimcp_result_t glial_integration_set_astrocyte_network(glial_integration_t* gi,
                                                       astrocyte_network_t* astrocyte_network) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);
    gi->astrocyte_network = astrocyte_network;
    nimcp_mutex_unlock(&gi->lock);

    return NIMCP_SUCCESS;
}

nimcp_result_t glial_integration_set_oligodendrocyte_network(
    glial_integration_t* gi, oligodendrocyte_network_t* oligo_network) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);
    gi->oligodendrocyte_network = oligo_network;
    nimcp_mutex_unlock(&gi->lock);

    return NIMCP_SUCCESS;
}

nimcp_result_t glial_integration_set_microglia_network(glial_integration_t* gi,
                                                       microglia_network_t* microglia_network) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);
    gi->microglia_network = microglia_network;
    nimcp_mutex_unlock(&gi->lock);

    return NIMCP_SUCCESS;
}

// ============================================================================
// GLIAL CELL ASSIGNMENT
// ============================================================================

nimcp_result_t glial_integration_assign_astrocyte_to_synapse(glial_integration_t* gi,
                                                             uint32_t astrocyte_id,
                                                             uint32_t synapse_id) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);

    // Forward mapping: synapse → astrocyte
    // Store the astrocyte_id directly (not a pointer to it)
    hash_table_insert_uint32(gi->synapse_to_astrocyte, synapse_id, &astrocyte_id, sizeof(uint32_t));

    // Reverse mapping: astrocyte → list of synapses
    id_list_t** list_ptr = (id_list_t**)hash_table_lookup_uint32(gi->astrocyte_to_synapses, astrocyte_id);
    id_list_t* list;

    if (!list_ptr) {
        // Create new list for this astrocyte
        list = id_list_create();
        if (!list) {
            nimcp_mutex_unlock(&gi->lock);
            return NIMCP_ERROR_MEMORY;
        }

        // Store pointer to list in hash table (hash table will copy the pointer value)
        hash_table_insert_uint32(gi->astrocyte_to_synapses, astrocyte_id, &list, sizeof(id_list_t*));
    } else {
        list = *list_ptr;
    }

    // Add synapse to list
    if (!id_list_add(list, synapse_id)) {
        nimcp_mutex_unlock(&gi->lock);
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_mutex_unlock(&gi->lock);
    return NIMCP_SUCCESS;
}

nimcp_result_t glial_integration_assign_oligodendrocyte_to_neuron(glial_integration_t* gi,
                                                                  uint32_t oligo_id,
                                                                  uint32_t neuron_id) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);

    // Forward mapping: neuron → oligodendrocyte
    // Store the oligo_id directly (not a pointer to it)
    hash_table_insert_uint32(gi->neuron_to_oligodendrocyte, neuron_id, &oligo_id, sizeof(uint32_t));

    // Reverse mapping: oligodendrocyte → list of neurons
    id_list_t** list_ptr = (id_list_t**)hash_table_lookup_uint32(gi->oligodendrocyte_to_neurons, oligo_id);
    id_list_t* list;

    if (!list_ptr) {
        list = id_list_create();
        if (!list) {
            nimcp_mutex_unlock(&gi->lock);
            return NIMCP_ERROR_MEMORY;
        }

        // Store pointer to list in hash table (hash table will copy the pointer value)
        hash_table_insert_uint32(gi->oligodendrocyte_to_neurons, oligo_id, &list, sizeof(id_list_t*));
    } else {
        list = *list_ptr;
    }

    if (!id_list_add(list, neuron_id)) {
        nimcp_mutex_unlock(&gi->lock);
        return NIMCP_ERROR_MEMORY;
    }

    nimcp_mutex_unlock(&gi->lock);
    return NIMCP_SUCCESS;
}

nimcp_result_t glial_integration_assign_microglia_to_synapse(glial_integration_t* gi,
                                                             uint32_t microglia_id,
                                                             uint32_t synapse_id) {
    if (!gi) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(&gi->lock);

    // Forward mapping: synapse → microglia
    // Store the microglia_id directly (not a pointer to it)
    hash_table_insert_uint32(gi->synapse_to_microglia, synapse_id, &microglia_id, sizeof(uint32_t));

    // Reverse mapping: microglia → list of synapses
    id_list_t** list_ptr = (id_list_t**)hash_table_lookup_uint32(gi->microglia_to_synapses, microglia_id);
    id_list_t* list;

    if (!list_ptr) {
        list = id_list_create();
        if (!list) {
            nimcp_mutex_unlock(&gi->lock);
            return NIMCP_ERROR_MEMORY;
        }

        // Store pointer to list in hash table (hash table will copy the pointer value)
        hash_table_insert_uint32(gi->microglia_to_synapses, microglia_id, &list, sizeof(id_list_t*));
    } else {
        list = *list_ptr;
    }

    if (!id_list_add(list, synapse_id)) {
        nimcp_mutex_unlock(&gi->lock);
        return NIMCP_ERROR_MEMORY;
    }

    // Also add to microglia's monitored synapse list
    if (gi->microglia_network && microglia_id < gi->microglia_network->num_microglia) {
        microglia_t* mg = gi->microglia_network->microglia[microglia_id];
        if (mg) {
            microglia_monitor_synapse(mg, synapse_id);
        }
    }

    nimcp_mutex_unlock(&gi->lock);
    return NIMCP_SUCCESS;
}

// ============================================================================
// AUTOMATIC SPATIAL ASSIGNMENT
// ============================================================================

uint32_t glial_integration_auto_assign_spatial(glial_integration_t* gi) {
    if (!gi) return 0;

    // TODO: Implement spatial assignment based on coordinates
    return 0;
}

// ============================================================================
// EVENT NOTIFICATIONS
// ============================================================================

void glial_integration_on_synapse_fired(glial_integration_t* gi, uint32_t pre_neuron_id,
                                        uint32_t post_neuron_id, float synaptic_weight,
                                        uint64_t timestamp) {
    if (!gi) return;

    uint32_t synapse_id = make_synapse_id(pre_neuron_id, post_neuron_id);

    nimcp_mutex_lock(&gi->lock);

    // Find assigned astrocyte
    if (gi->enable_astrocyte_modulation && gi->astrocyte_network) {
        uint32_t* astrocyte_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->synapse_to_astrocyte, synapse_id);

        if (astrocyte_id_ptr) {
            uint32_t astrocyte_id = *astrocyte_id_ptr;
            if (astrocyte_id < gi->astrocyte_network->num_astrocytes) {
                astrocyte_t* ast = gi->astrocyte_network->astrocytes[astrocyte_id];
                if (ast) {
                    // Increase calcium with external stimulus
                    // dt = 0.001s (1ms), stimulus = synaptic_weight * 10
                    astrocyte_update_calcium(ast, 0.001f, synaptic_weight * 10.0f);
                }
            }
        }
    }

    // Notify microglia if assigned
    if (gi->enable_microglia_pruning && gi->microglia_network) {
        uint32_t* microglia_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->synapse_to_microglia, synapse_id);

        if (microglia_id_ptr) {
            uint32_t microglia_id = *microglia_id_ptr;
            if (microglia_id < gi->microglia_network->num_microglia) {
                microglia_t* mg = gi->microglia_network->microglia[microglia_id];
                if (mg) {
                    // Track synapse activity
                    microglia_track_synapse_activity(mg, synapse_id, synaptic_weight, timestamp);
                }
            }
        }
    }

    nimcp_mutex_unlock(&gi->lock);
}

void glial_integration_on_neuron_fired(glial_integration_t* gi, uint32_t neuron_id,
                                       uint64_t timestamp) {
    if (!gi || !gi->enable_oligodendrocyte_myelination) return;

    nimcp_mutex_lock(&gi->lock);

    // Find assigned oligodendrocyte
    if (gi->oligodendrocyte_network) {
        uint32_t* oligo_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->neuron_to_oligodendrocyte, neuron_id);

        if (oligo_id_ptr) {
            uint32_t oligo_id = *oligo_id_ptr;
            if (oligo_id < gi->oligodendrocyte_network->num_oligodendrocytes) {
                oligodendrocyte_t* oligo = gi->oligodendrocyte_network->oligodendrocytes[oligo_id];
                if (oligo) {
                    // Track axon activity for adaptive myelination
                    oligodendrocyte_track_activity(oligo, neuron_id, 1.0f, timestamp);
                }
            }
        }
    }

    nimcp_mutex_unlock(&gi->lock);
}

// ============================================================================
// GLIAL MODULATION QUERIES
// ============================================================================

float glial_integration_get_synaptic_modulation(glial_integration_t* gi, uint32_t pre_neuron_id,
                                                 uint32_t post_neuron_id) {
    if (!gi || !gi->enable_astrocyte_modulation) {
        return 1.0f; // Neutral modulation
    }

    uint32_t synapse_id = make_synapse_id(pre_neuron_id, post_neuron_id);

    nimcp_mutex_lock(&gi->lock);

    // Find assigned astrocyte
    if (gi->astrocyte_network) {
        uint32_t* astrocyte_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->synapse_to_astrocyte, synapse_id);

        if (astrocyte_id_ptr) {
            uint32_t astrocyte_id = *astrocyte_id_ptr;
            if (astrocyte_id < gi->astrocyte_network->num_astrocytes) {
                astrocyte_t* ast = gi->astrocyte_network->astrocytes[astrocyte_id];
                if (ast) {
                    // Get modulation factor based on glutamate pool
                    float glutamate = ast->glutamate_pool;
                    float modulation = 1.0f + (glutamate / 100.0f); // Simple linear mapping

                    // Clamp to range
                    if (modulation < 0.8f) modulation = 0.8f;
                    if (modulation > 1.2f) modulation = 1.2f;

                    nimcp_mutex_unlock(&gi->lock);
                    return modulation;
                }
            }
        }
    }

    nimcp_mutex_unlock(&gi->lock);
    return 1.0f; // No astrocyte → neutral
}

float glial_integration_get_myelination_factor(glial_integration_t* gi, uint32_t neuron_id) {
    if (!gi || !gi->enable_oligodendrocyte_myelination) {
        return 0.0f; // No myelination
    }

    nimcp_mutex_lock(&gi->lock);

    // Find assigned oligodendrocyte
    if (gi->oligodendrocyte_network) {
        uint32_t* oligo_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->neuron_to_oligodendrocyte, neuron_id);

        if (oligo_id_ptr) {
            uint32_t oligo_id = *oligo_id_ptr;
            if (oligo_id < gi->oligodendrocyte_network->num_oligodendrocytes) {
                oligodendrocyte_t* oligo = gi->oligodendrocyte_network->oligodendrocytes[oligo_id];
                if (oligo) {
                    // Get myelination level for this neuron/axon
                    float myelination = oligodendrocyte_get_myelination_level(oligo, neuron_id);
                    nimcp_mutex_unlock(&gi->lock);
                    return myelination;
                }
            }
        }
    }

    nimcp_mutex_unlock(&gi->lock);
    return 0.0f; // No oligodendrocyte
}

bool glial_integration_should_prune_synapse(glial_integration_t* gi, uint32_t pre_neuron_id,
                                            uint32_t post_neuron_id) {
    if (!gi || !gi->enable_microglia_pruning) {
        return false;
    }

    uint32_t synapse_id = make_synapse_id(pre_neuron_id, post_neuron_id);

    nimcp_mutex_lock(&gi->lock);

    // Find assigned microglia
    if (gi->microglia_network) {
        uint32_t* microglia_id_ptr = (uint32_t*)hash_table_lookup_uint32(
            gi->synapse_to_microglia, synapse_id);

        if (microglia_id_ptr) {
            uint32_t microglia_id = *microglia_id_ptr;
            if (microglia_id < gi->microglia_network->num_microglia) {
                microglia_t* mg = gi->microglia_network->microglia[microglia_id];
                if (mg) {
                    // Check if activity score is below pruning threshold
                    float activity_score = microglia_get_synapse_activity_score(mg, synapse_id);
                    bool should_prune = (activity_score < mg->pruning_threshold);
                    nimcp_mutex_unlock(&gi->lock);
                    return should_prune;
                }
            }
        }
    }

    nimcp_mutex_unlock(&gi->lock);
    return false;
}

// ============================================================================
// SIMULATION STEP
// ============================================================================

void glial_integration_step(glial_integration_t* gi, uint64_t timestamp) {
    if (!gi) return;

    nimcp_mutex_lock(&gi->lock);

    // Step astrocyte network (calcium dynamics, glutamate release)
    if (gi->astrocyte_network && gi->enable_astrocyte_modulation) {
        astrocyte_network_step(gi->astrocyte_network, timestamp);
    }

    // Step oligodendrocyte network (adaptive myelination)
    if (gi->oligodendrocyte_network && gi->enable_oligodendrocyte_myelination) {
        oligodendrocyte_network_step(gi->oligodendrocyte_network, timestamp);
    }

    // Step microglia network (activity scoring, pruning)
    if (gi->microglia_network && gi->enable_microglia_pruning) {
        microglia_network_step(gi->microglia_network, timestamp);
    }

    nimcp_mutex_unlock(&gi->lock);
}

// ============================================================================
// STATISTICS & MONITORING
// ============================================================================

nimcp_result_t glial_integration_get_stats(glial_integration_t* gi,
                                           glial_integration_stats_t* stats) {
    if (!gi || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(stats, 0, sizeof(glial_integration_stats_t));

    nimcp_mutex_lock(&gi->lock);

    // Count glial cells
    if (gi->astrocyte_network) {
        stats->num_astrocytes = gi->astrocyte_network->num_astrocytes;
    }
    if (gi->oligodendrocyte_network) {
        stats->num_oligodendrocytes = gi->oligodendrocyte_network->num_oligodendrocytes;
    }
    if (gi->microglia_network) {
        stats->num_microglia = gi->microglia_network->num_microglia;
    }

    // Count assignments from forward mappings
    stats->num_tripartite_synapses = (uint32_t)hash_table_size(gi->synapse_to_astrocyte);
    stats->num_myelinated_neurons = (uint32_t)hash_table_size(gi->neuron_to_oligodendrocyte);
    stats->num_monitored_synapses = (uint32_t)hash_table_size(gi->synapse_to_microglia);

    // Copy statistics
    stats->total_modulations = gi->total_astrocyte_modulations;
    stats->total_myelinations = gi->total_oligodendrocyte_myelinations;
    stats->total_prunings = gi->total_microglia_prunings;

    nimcp_mutex_unlock(&gi->lock);

    return NIMCP_SUCCESS;
}

uint32_t glial_integration_get_astrocyte_synapse_count(glial_integration_t* gi,
                                                       uint32_t astrocyte_id) {
    if (!gi) return 0;

    nimcp_mutex_lock(&gi->lock);

    id_list_t** list_ptr = (id_list_t**)hash_table_lookup_uint32(gi->astrocyte_to_synapses, astrocyte_id);
    uint32_t count = 0;

    if (list_ptr && *list_ptr) {
        count = (*list_ptr)->count;
    }

    nimcp_mutex_unlock(&gi->lock);
    return count;
}

uint32_t glial_integration_get_oligodendrocyte_neuron_count(glial_integration_t* gi,
                                                            uint32_t oligo_id) {
    if (!gi) return 0;

    nimcp_mutex_lock(&gi->lock);

    id_list_t** list_ptr = (id_list_t**)hash_table_lookup_uint32(gi->oligodendrocyte_to_neurons, oligo_id);
    uint32_t count = 0;

    if (list_ptr && *list_ptr) {
        count = (*list_ptr)->count;
    }

    nimcp_mutex_unlock(&gi->lock);
    return count;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

void glial_integration_set_astrocyte_modulation_enabled(glial_integration_t* gi, bool enable) {
    if (!gi) return;

    nimcp_mutex_lock(&gi->lock);
    gi->enable_astrocyte_modulation = enable;
    nimcp_mutex_unlock(&gi->lock);
}

void glial_integration_set_oligodendrocyte_myelination_enabled(glial_integration_t* gi,
                                                               bool enable) {
    if (!gi) return;

    nimcp_mutex_lock(&gi->lock);
    gi->enable_oligodendrocyte_myelination = enable;
    nimcp_mutex_unlock(&gi->lock);
}

void glial_integration_set_microglia_pruning_enabled(glial_integration_t* gi, bool enable) {
    if (!gi) return;

    nimcp_mutex_lock(&gi->lock);
    gi->enable_microglia_pruning = enable;
    nimcp_mutex_unlock(&gi->lock);
}
