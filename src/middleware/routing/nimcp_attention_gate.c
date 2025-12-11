//=============================================================================
// nimcp_attention_gate.c - Attention Mechanism for Routing
//=============================================================================

#include "middleware/routing/nimcp_attention_gate.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"



#define LOG_MODULE "nimcp_attention_gate"
#define LOG_MODULE_ID 0x0529

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

#define MAX_SHIFTS_HISTORY 100

typedef struct {
    uint32_t source_id;
    uint32_t target_id;
    attention_target_t target;
} attention_entry_t;

// Internal structure for sorting in update_spotlight (8 bytes)
typedef struct {
    uint32_t target_id;
    float weight;
} weighted_target_t;

struct attention_gate {
    // Configuration
    attention_gate_config_t config;

    // Attention targets (hash table)
    attention_entry_t* entries;
    uint32_t num_entries;
    uint32_t capacity;

    // Attention spotlight
    uint32_t* spotlight_ids;
    uint32_t num_in_spotlight;

    // Shift history
    attention_shift_t* shift_history;
    uint32_t num_shifts;
    uint32_t shift_capacity;

    // Memory pool for hot-path allocations (Phase 1.5)
    // Pool for sorting buffer in update_spotlight - max_targets weighted_target_t
    memory_pool_t sort_buffer_pool;

    // Statistics
    uint64_t total_shifts;
    uint32_t current_winner;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static uint64_t make_key(uint32_t source_id, uint32_t target_id) {
    return ((uint64_t)source_id << 32) | (uint64_t)target_id;
}

static attention_entry_t* find_entry(const attention_gate_t* gate,
                                     uint32_t source_id,
                                     uint32_t target_id) {
    for (uint32_t i = 0; i < gate->num_entries; i++) {
        if (gate->entries[i].source_id == source_id &&
            gate->entries[i].target_id == target_id) {
            return &gate->entries[i];
        }
    }
    return NULL;
}

static bool add_entry(attention_gate_t* gate, uint32_t source_id, uint32_t target_id) {
    if (gate->num_entries >= gate->capacity) {
        return false;
    }

    attention_entry_t* entry = &gate->entries[gate->num_entries];
    entry->source_id = source_id;
    entry->target_id = target_id;

    entry->target.target_id = target_id;
    entry->target.topdown_weight = 0.0F;
    entry->target.bottomup_salience = 0.0F;
    entry->target.combined_weight = 0.0F;
    entry->target.in_spotlight = false;
    entry->target.last_update_ms = 0;

    gate->num_entries++;

    return true;
}

static void update_combined_weight(attention_gate_t* gate, attention_entry_t* entry) {
    float topdown = entry->target.topdown_weight * gate->config.topdown_weight;
    float bottomup = entry->target.bottomup_salience * gate->config.bottomup_weight;

    switch (gate->config.mode) {
        case ATTENTION_MODE_TOPDOWN:
            entry->target.combined_weight = entry->target.topdown_weight;
            break;
        case ATTENTION_MODE_BOTTOMUP:
            entry->target.combined_weight = entry->target.bottomup_salience;
            break;
        case ATTENTION_MODE_MIXED:
            entry->target.combined_weight = topdown + bottomup;
            if (entry->target.combined_weight > 1.0F) {
                entry->target.combined_weight = 1.0F;
            }
            break;
    }
}

static void record_shift(attention_gate_t* gate, uint32_t from_target,
                        uint32_t to_target, float magnitude) {
    if (!gate->config.enable_shift_detection) return;

    if (gate->num_shifts < gate->shift_capacity) {
        attention_shift_t* shift = &gate->shift_history[gate->num_shifts];
        shift->from_target = from_target;
        shift->to_target = to_target;
        shift->shift_time_ms = 0.0;  // Placeholder
        shift->shift_magnitude = magnitude;
        gate->num_shifts++;
    } else {
        // Circular buffer
        for (uint32_t i = 0; i < gate->shift_capacity - 1; i++) {
            gate->shift_history[i] = gate->shift_history[i + 1];
        }
        attention_shift_t* shift = &gate->shift_history[gate->shift_capacity - 1];
        shift->from_target = from_target;
        shift->to_target = to_target;
        shift->shift_time_ms = 0.0;
        shift->shift_magnitude = magnitude;
    }

    gate->total_shifts++;
}

// ============================================================================
// PUBLIC API
// ============================================================================

attention_gate_config_t attention_gate_default_config(void) {
    attention_gate_config_t config;
    config.mode = ATTENTION_MODE_MIXED;
    config.max_targets = ATTENTION_MAX_TARGETS;
    config.spotlight_size = ATTENTION_SPOTLIGHT_SIZE;
    config.enable_winner_take_all = false;
    config.enable_shift_detection = true;
    config.topdown_weight = 0.7F;
    config.bottomup_weight = 0.3F;
    config.inhibition_strength = 0.5F;
    return config;
}

attention_gate_t* attention_gate_create(const attention_gate_config_t* config) {
    if (!config || config->max_targets == 0) {
        return NULL;
    }

    attention_gate_t* gate = (attention_gate_t*)nimcp_calloc(1, sizeof(attention_gate_t));
    if (!gate) return NULL;

    gate->config = *config;

    // Allocate entries
    gate->capacity = config->max_targets;
    gate->entries = (attention_entry_t*)nimcp_calloc(gate->capacity,
                                               sizeof(attention_entry_t));
    if (!gate->entries) {
        attention_gate_destroy(gate);
        return NULL;
    }

    gate->num_entries = 0;

    // Allocate spotlight
    gate->spotlight_ids = (uint32_t*)nimcp_calloc(config->spotlight_size, sizeof(uint32_t));
    if (!gate->spotlight_ids) {
        attention_gate_destroy(gate);
        return NULL;
    }

    gate->num_in_spotlight = 0;

    // Allocate shift history
    if (config->enable_shift_detection) {
        gate->shift_capacity = MAX_SHIFTS_HISTORY;
        gate->shift_history = (attention_shift_t*)nimcp_calloc(gate->shift_capacity,
                                                         sizeof(attention_shift_t));
        if (!gate->shift_history) {
            attention_gate_destroy(gate);
            return NULL;
        }
    }

    // Initialize memory pool for hot-path allocations (Phase 1.5)
    // Pool for sorting buffer in update_spotlight - 2 blocks for concurrent calls
    memory_pool_config_t sort_pool_config = {
        .block_size = config->max_targets * sizeof(weighted_target_t),
        .num_blocks = 2,
        .alignment = 16,  // SIMD alignment
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    gate->sort_buffer_pool = memory_pool_create(&sort_pool_config);

    if (!gate->sort_buffer_pool) {
        attention_gate_destroy(gate);
        return NULL;
    }

    gate->num_shifts = 0;
    gate->total_shifts = 0;
    gate->current_winner = 0;

    return gate;
}

void attention_gate_destroy(attention_gate_t* gate) {
    if (!gate) return;

    nimcp_free(gate->entries);
    nimcp_free(gate->spotlight_ids);
    nimcp_free(gate->shift_history);

    // Destroy memory pool (Phase 1.5)
    memory_pool_destroy(gate->sort_buffer_pool);
    nimcp_free(gate);
}

bool attention_gate_set_weight(attention_gate_t* gate,
                                uint32_t source_id,
                                uint32_t target_id,
                                float weight) {
    if (!gate || weight < 0.0F || weight > 1.0F) {
        return false;
    }

    attention_entry_t* entry = find_entry(gate, source_id, target_id);

    if (!entry) {
        if (!add_entry(gate, source_id, target_id)) {
            return false;
        }
        entry = find_entry(gate, source_id, target_id);
    }

    if (!entry) return false;

    float old_weight = entry->target.combined_weight;
    entry->target.topdown_weight = weight;

    update_combined_weight(gate, entry);

    // Detect significant shift
    float shift = fabsf(entry->target.combined_weight - old_weight);
    if (shift >= ATTENTION_SHIFT_THRESHOLD) {
        uint32_t old_winner = gate->current_winner;
        gate->current_winner = target_id;

        if (old_winner != target_id) {
            record_shift(gate, old_winner, target_id, shift);
        }
    }

    return true;
}

bool attention_gate_get_weight(const attention_gate_t* gate,
                                uint32_t source_id,
                                uint32_t target_id,
                                float* weight) {
    if (!gate || !weight) return false;

    const attention_entry_t* entry = find_entry(gate, source_id, target_id);

    if (!entry) {
        *weight = 0.0F;
        return true;  // Return true with 0.0 weight - this is a valid state
    }

    *weight = entry->target.combined_weight;
    return true;
}

bool attention_gate_update_salience(attention_gate_t* gate,
                                     uint32_t target_id,
                                     float salience) {
    if (!gate || salience < 0.0F || salience > 1.0F) {
        return false;
    }

    // Update all entries with this target
    bool updated = false;

    for (uint32_t i = 0; i < gate->num_entries; i++) {
        if (gate->entries[i].target_id == target_id) {
            gate->entries[i].target.bottomup_salience = salience;
            update_combined_weight(gate, &gate->entries[i]);
            updated = true;
        }
    }

    // If no entry exists, create one with source_id = 0 (default)
    if (!updated) {
        if (!add_entry(gate, 0, target_id)) {
            return false;
        }

        // Find the newly added entry and update it
        attention_entry_t* entry = find_entry(gate, 0, target_id);
        if (entry) {
            entry->target.bottomup_salience = salience;
            update_combined_weight(gate, entry);
            updated = true;
        }
    }

    return updated;
}

bool attention_gate_apply_wta(attention_gate_t* gate, uint32_t* winner_id) {
    if (!gate || gate->num_entries == 0) {
        return false;
    }

    // Find maximum weight
    float max_weight = 0.0F;
    uint32_t winner_idx = 0;

    for (uint32_t i = 0; i < gate->num_entries; i++) {
        if (gate->entries[i].target.combined_weight > max_weight) {
            max_weight = gate->entries[i].target.combined_weight;
            winner_idx = i;
        }
    }

    // Set winner to 1.0, suppress all others
    for (uint32_t i = 0; i < gate->num_entries; i++) {
        if (i == winner_idx) {
            gate->entries[i].target.combined_weight = 1.0F;
        } else {
            // Apply lateral inhibition
            float inhibited = gate->entries[i].target.combined_weight *
                            (1.0F - gate->config.inhibition_strength);
            gate->entries[i].target.combined_weight = inhibited;
        }
    }

    if (winner_id) {
        *winner_id = gate->entries[winner_idx].target_id;
    }

    uint32_t old_winner = gate->current_winner;
    gate->current_winner = gate->entries[winner_idx].target_id;

    if (old_winner != gate->current_winner) {
        record_shift(gate, old_winner, gate->current_winner, 1.0F);
    }

    return true;
}

bool attention_gate_update_spotlight(attention_gate_t* gate,
                                      uint32_t* spotlight_ids,
                                      uint32_t* num_in_spotlight) {
    if (!gate) return false;

    // Sort entries by combined weight - Phase 1.5 O(1) pool allocation
    weighted_target_t* targets = (weighted_target_t*)memory_pool_acquire(gate->sort_buffer_pool);
    if (!targets) return false;

    for (uint32_t i = 0; i < gate->num_entries; i++) {
        targets[i].target_id = gate->entries[i].target_id;
        targets[i].weight = gate->entries[i].target.combined_weight;
    }

    // Sort descending by weight (bubble sort for simplicity)
    for (uint32_t i = 0; i < gate->num_entries; i++) {
        for (uint32_t j = i + 1; j < gate->num_entries; j++) {
            if (targets[j].weight > targets[i].weight) {
                weighted_target_t temp = targets[i];
                targets[i] = targets[j];
                targets[j] = temp;
            }
        }
    }

    // Select top N for spotlight
    uint32_t spotlight_count = (gate->num_entries < gate->config.spotlight_size) ?
                              gate->num_entries : gate->config.spotlight_size;

    // Clear spotlight flags
    for (uint32_t i = 0; i < gate->num_entries; i++) {
        gate->entries[i].target.in_spotlight = false;
    }

    // Set spotlight
    for (uint32_t i = 0; i < spotlight_count; i++) {
        gate->spotlight_ids[i] = targets[i].target_id;

        // Mark in spotlight
        for (uint32_t j = 0; j < gate->num_entries; j++) {
            if (gate->entries[j].target_id == targets[i].target_id) {
                gate->entries[j].target.in_spotlight = true;
                break;
            }
        }
    }

    gate->num_in_spotlight = spotlight_count;

    if (spotlight_ids) {
        memcpy(spotlight_ids, gate->spotlight_ids,
               spotlight_count * sizeof(uint32_t));
    }

    if (num_in_spotlight) {
        *num_in_spotlight = spotlight_count;
    }

    // Release back to pool (Phase 1.5)
    memory_pool_release(gate->sort_buffer_pool, targets);

    return true;
}

bool attention_gate_get_shifts(const attention_gate_t* gate,
                                attention_shift_t* shifts,
                                uint32_t max_shifts,
                                uint32_t* num_shifts) {
    if (!gate || !shifts || !num_shifts) {
        return false;
    }

    uint32_t count = (gate->num_shifts < max_shifts) ?
                    gate->num_shifts : max_shifts;

    memcpy(shifts, gate->shift_history, count * sizeof(attention_shift_t));
    *num_shifts = count;

    return true;
}

void attention_gate_reset(attention_gate_t* gate) {
    if (!gate) return;

    for (uint32_t i = 0; i < gate->num_entries; i++) {
        gate->entries[i].target.topdown_weight = 0.0F;
        gate->entries[i].target.bottomup_salience = 0.0F;
        gate->entries[i].target.combined_weight = 0.0F;
        gate->entries[i].target.in_spotlight = false;
    }

    gate->num_in_spotlight = 0;
    gate->num_shifts = 0;
    gate->current_winner = 0;
}

bool attention_gate_get_stats(const attention_gate_t* gate,
                               uint32_t* num_targets,
                               uint32_t* num_in_spotlight,
                               uint64_t* total_shifts) {
    if (!gate) return false;

    if (num_targets) *num_targets = gate->num_entries;
    if (num_in_spotlight) *num_in_spotlight = gate->num_in_spotlight;
    if (total_shifts) *total_shifts = gate->total_shifts;

    return true;
}
