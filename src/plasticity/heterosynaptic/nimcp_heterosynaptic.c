/**
 * @file nimcp_heterosynaptic.c
 * @brief Heterosynaptic Plasticity Implementation
 * @version 1.0.0
 * @date 2025-12-19
 */

#include "plasticity/heterosynaptic/nimcp_heterosynaptic.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/validation/nimcp_common.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * WHAT: Create spatial index for fast neighbor lookup
 * WHY:  O(1) average case vs O(n) linear search
 * HOW:  Allocate 3D grid with spatial hashing
 */
static hetero_spatial_index_t* create_spatial_index(
    const hetero_config_t* config,
    const float world_min[3],
    const float world_max[3])
{
    if (!config) return NULL;

    hetero_spatial_index_t* index = nimcp_malloc(sizeof(hetero_spatial_index_t));
    if (!index) return NULL;

    index->grid_size = config->spatial_grid_size;

    /* CRITICAL: Overflow check for grid_size^3
     * WHY:  grid_size * grid_size * grid_size can overflow uint32_t
     * FIX:  Limit grid_size to prevent overflow (1000^3 = 1B > UINT32_MAX/2)
     */
    if (index->grid_size > 1000) {
        NIMCP_LOGGING_ERROR("Spatial grid size %u exceeds maximum (1000)",
                            index->grid_size);
        nimcp_free(index);
        return NULL;
    }

    index->cell_width = (world_max[0] - world_min[0]) / index->grid_size;

    /* Copy world bounds */
    memcpy(index->world_min, world_min, 3 * sizeof(float));
    memcpy(index->world_max, world_max, 3 * sizeof(float));

    /* Allocate grid - 3D array of synapse lists */
    size_t total_cells = (size_t)index->grid_size * index->grid_size * index->grid_size;
    index->grid = nimcp_malloc(total_cells * sizeof(hetero_synapse_t**));
    index->cell_counts = nimcp_calloc(total_cells, sizeof(uint32_t));
    index->cell_capacities = nimcp_calloc(total_cells, sizeof(uint32_t));

    if (!index->grid || !index->cell_counts || !index->cell_capacities) {
        nimcp_free(index->grid);
        nimcp_free(index->cell_counts);
        nimcp_free(index->cell_capacities);
        nimcp_free(index);
        return NULL;
    }

    return index;
}

/**
 * WHAT: Destroy spatial index
 * WHY:  Free all allocated memory
 * HOW:  Free grid cells and index structure
 */
static void destroy_spatial_index(hetero_spatial_index_t* index) {
    if (!index) return;

    if (index->grid) {
        size_t total_cells = index->grid_size * index->grid_size * index->grid_size;
        for (size_t i = 0; i < total_cells; i++) {
            nimcp_free(index->grid[i]);
        }
        nimcp_free(index->grid);
    }

    nimcp_free(index->cell_counts);
    nimcp_free(index->cell_capacities);
    nimcp_free(index);
}

/**
 * WHAT: Compute spatial hash grid cell index
 * WHY:  Map 3D position to grid cell
 * HOW:  Discretize position and compute linear index
 */
static uint32_t compute_grid_cell(
    const hetero_spatial_index_t* index,
    const hetero_spatial_coords_t* pos)
{
    /* Clamp to world bounds */
    float x = fmaxf(index->world_min[0], fminf(index->world_max[0], pos->x));
    float y = fmaxf(index->world_min[1], fminf(index->world_max[1], pos->y));
    float z = fmaxf(index->world_min[2], fminf(index->world_max[2], pos->z));

    /* Compute grid coordinates */
    uint32_t gx = (uint32_t)((x - index->world_min[0]) / index->cell_width);
    uint32_t gy = (uint32_t)((y - index->world_min[1]) / index->cell_width);
    uint32_t gz = (uint32_t)((z - index->world_min[2]) / index->cell_width);

    /* Clamp to grid size */
    gx = gx >= index->grid_size ? index->grid_size - 1 : gx;
    gy = gy >= index->grid_size ? index->grid_size - 1 : gy;
    gz = gz >= index->grid_size ? index->grid_size - 1 : gz;

    /* Linear index */
    return gz * index->grid_size * index->grid_size + gy * index->grid_size + gx;
}

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int hetero_default_config(hetero_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    config->neighbor_radius = HETERO_DEFAULT_NEIGHBOR_RADIUS;
    config->depression_factor = HETERO_DEFAULT_DEPRESSION_FACTOR;
    config->decay_lambda = HETERO_DEFAULT_DECAY_LAMBDA;
    config->delay_ms = HETERO_DEFAULT_DELAY_MS;

    config->enable_competition = true;
    config->wta_threshold = HETERO_WTA_THRESHOLD;
    config->wta_suppression_factor = HETERO_WTA_SUPPRESSION_FACTOR;

    config->enable_sleep_modulation = true;
    config->enable_immune_modulation = false;
    config->enable_bio_async = false;

    config->enable_spatial_index = true;
    config->spatial_grid_size = HETERO_SPATIAL_GRID_CELLS;

    config->on_competition_event = NULL;
    config->callback_user_data = NULL;

    return 0;
}

hetero_system_t* hetero_create(const hetero_config_t* config, size_t initial_capacity) {
    /* Use defaults if no config provided */
    hetero_config_t default_config;
    if (!config) {
        hetero_default_config(&default_config);
        config = &default_config;
    }

    /* Allocate system */
    hetero_system_t* system = nimcp_malloc(sizeof(hetero_system_t));
    if (!system) {
        NIMCP_LOGGING_ERROR("Failed to allocate heterosynaptic system");
        return NULL;
    }

    /* Initialize fields */
    memcpy(&system->config, config, sizeof(hetero_config_t));
    system->synapses = NULL;
    system->num_synapses = 0;
    system->synapse_capacity = 0;
    system->spatial_index = NULL;
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;

    /* Allocate synapse array */
    if (initial_capacity > 0) {
        system->synapses = nimcp_malloc(initial_capacity * sizeof(hetero_synapse_t));
        if (!system->synapses) {
            NIMCP_LOGGING_ERROR("Failed to allocate synapse array");
            nimcp_free(system);
            return NULL;
        }
        system->synapse_capacity = initial_capacity;
    }

    /* Create spatial index if enabled */
    if (config->enable_spatial_index) {
        float world_min[3] = {-100.0f, -100.0f, -100.0f};
        float world_max[3] = {100.0f, 100.0f, 100.0f};
        system->spatial_index = create_spatial_index(config, world_min, world_max);
        if (!system->spatial_index) {
            NIMCP_LOGGING_WARN("Failed to create spatial index, using linear search");
        }
    }

    /* Create mutex */
    system->mutex = nimcp_platform_mutex_create();
    if (!system->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        destroy_spatial_index(system->spatial_index);
        nimcp_free(system->synapses);
        nimcp_free(system);
        return NULL;
    }

    /* Reset statistics */
    system->total_competitions = 0;
    system->total_depressions = 0;
    system->avg_neighbors_per_competition = 0.0f;

    NIMCP_LOGGING_INFO("Created heterosynaptic system with capacity %zu", initial_capacity);
    return system;
}

void hetero_destroy(hetero_system_t* system) {
    if (!system) return;

    /* Destroy all synapse mutexes (heap-allocated for realloc safety) */
    if (system->synapses) {
        for (size_t i = 0; i < system->num_synapses; i++) {
            if (system->synapses[i].lock) {
                nimcp_platform_mutex_destroy(system->synapses[i].lock);
                nimcp_free(system->synapses[i].lock);
                system->synapses[i].lock = NULL;
            }
        }
        nimcp_free(system->synapses);
        system->synapses = NULL;
    }

    if (system->spatial_index) {
        destroy_spatial_index(system->spatial_index);
        system->spatial_index = NULL;
    }

    if (system->mutex) {
        nimcp_platform_mutex_destroy(system->mutex);
        system->mutex = NULL;
    }

    nimcp_free(system);

    NIMCP_LOGGING_INFO("Destroyed heterosynaptic system");
}

/* ============================================================================
 * Synapse Management API Implementation
 * ============================================================================ */

int hetero_add_synapse(
    hetero_system_t* system,
    const hetero_spatial_coords_t* position,
    float initial_weight,
    uint32_t synapse_id,
    uint32_t neuron_id)
{
    if (!system || !position) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(system->mutex);

    /* Resize if needed */
    if (system->num_synapses >= system->synapse_capacity) {
        size_t new_capacity = system->synapse_capacity == 0 ? 64 : system->synapse_capacity * 2;
        hetero_synapse_t* old_synapses = system->synapses;  /* Save for rollback */
        hetero_synapse_t* new_synapses = nimcp_realloc(
            system->synapses,
            new_capacity * sizeof(hetero_synapse_t)
        );
        if (!new_synapses) {
            system->synapses = old_synapses;  /* Restore on failure */
            nimcp_platform_mutex_unlock(system->mutex);
            NIMCP_LOGGING_ERROR("Failed to resize synapse array");
            return NIMCP_ERROR_NO_MEMORY;
        }
        system->synapses = new_synapses;
        system->synapse_capacity = new_capacity;

        /* NOTE: Existing synapses have heap-allocated locks (pointers) that
         * survive realloc since only the pointer is copied, not the mutex.
         * We only initialize the NEW synapse's lock below. */
    }

    /* Initialize the NEW synapse (at index num_synapses) */
    hetero_synapse_t* syn = &system->synapses[system->num_synapses];
    syn->weight = initial_weight;
    syn->w_max = 1.0f;
    syn->w_min = 0.0f;
    syn->position = *position;
    syn->synapse_id = synapse_id;
    syn->postsynaptic_neuron_id = neuron_id;
    syn->last_potentiation = 0.0f;
    syn->last_depression = 0.0f;
    syn->last_potentiation_time_ms = 0;
    syn->is_eligible_for_competition = true;
    syn->current_sleep_state = SLEEP_STATE_AWAKE;
    syn->num_competitions = 0;
    syn->num_wins = 0;
    syn->num_neighbor_depressions = 0;
    syn->total_hetero_ltd = 0.0f;

    /* Allocate lock on heap for realloc safety - pointer survives realloc */
    syn->lock = nimcp_platform_mutex_create();
    if (!syn->lock) {
        nimcp_platform_mutex_unlock(system->mutex);
        NIMCP_LOGGING_ERROR("Failed to create synapse mutex");
        return NIMCP_ERROR_NO_MEMORY;
    }

    system->num_synapses++;

    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

int hetero_remove_synapse(hetero_system_t* system, uint32_t synapse_id) {
    if (!system) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(system->mutex);

    /* Find synapse */
    size_t found_idx = (size_t)-1;
    for (size_t i = 0; i < system->num_synapses; i++) {
        if (system->synapses[i].synapse_id == synapse_id) {
            found_idx = i;
            break;
        }
    }

    if (found_idx == (size_t)-1) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Destroy and free heap-allocated mutex */
    if (system->synapses[found_idx].lock) {
        nimcp_platform_mutex_destroy(system->synapses[found_idx].lock);
        nimcp_free(system->synapses[found_idx].lock);
        system->synapses[found_idx].lock = NULL;
    }

    /* Shift remaining synapses */
    if (found_idx < system->num_synapses - 1) {
        memmove(&system->synapses[found_idx],
                &system->synapses[found_idx + 1],
                (system->num_synapses - found_idx - 1) * sizeof(hetero_synapse_t));
    }

    system->num_synapses--;
    nimcp_platform_mutex_unlock(system->mutex);
    return 0;
}

/**
 * WHAT: Get synapse by ID
 * WHY:  Locate synapse in array for direct manipulation
 * HOW:  Linear search with mutex protection
 *
 * @deprecated This function is DEPRECATED for multi-threaded code.
 *             Use hetero_get_synapse_copy() instead for thread-safe access.
 *
 * CRITICAL THREAD SAFETY WARNING - POINTER INVALIDATION RISK:
 * ============================================================
 * The returned pointer points into the internal synapses array and becomes
 * INVALID if realloc or removal occurs. The mutex is released BEFORE return,
 * creating a race condition window.
 *
 * POINTER INVALIDATION SCENARIOS:
 *   1. Another thread calls hetero_add_synapse() and triggers realloc
 *   2. Another thread calls hetero_remove_synapse() on any synapse
 *   3. hetero_destroy() is called
 *
 * SAFE USAGE PATTERNS (caller MUST ensure one of these):
 *   - Single-threaded access to the hetero_system
 *   - Caller holds external synchronization preventing add/remove/destroy
 *   - Use individual synapse->lock for weight field modifications only
 *   - Copy needed data immediately after call, don't store pointer
 *
 * FOR THREAD-SAFE QUERIES: Use hetero_get_synapse_copy() instead, which
 * returns a copy of the synapse data that is safe to use across threads.
 *
 * The individual synapse->lock protects weight modifications but does NOT
 * protect against the pointer itself becoming invalid due to realloc/remove.
 *
 * RISK LEVEL: HIGH - Use with extreme caution in multi-threaded code.
 */
hetero_synapse_t* hetero_get_synapse(hetero_system_t* system, uint32_t synapse_id) {
    if (!system) return NULL;

    /* Lock during search to prevent concurrent modification during lookup.
     * CRITICAL: After unlock, returned pointer stability depends on caller
     * coordination - see warning above. Caller must immediately copy data
     * or hold external synchronization. */
    nimcp_platform_mutex_lock(system->mutex);

    hetero_synapse_t* result = NULL;
    for (size_t i = 0; i < system->num_synapses; i++) {
        if (system->synapses[i].synapse_id == synapse_id) {
            result = &system->synapses[i];
            break;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);

    /* WARNING: result pointer may be invalidated by concurrent operations.
     * Caller must use immediately or hold external synchronization. */
    return result;
}

/**
 * WHAT: Get a thread-safe copy of synapse data by ID
 * WHY:  hetero_get_synapse() returns pointer that can be invalidated by realloc
 * HOW:  Copy synapse data while holding mutex, return copy via out parameter
 *
 * This is the RECOMMENDED API for multi-threaded access.
 *
 * @param system Heterosynaptic system
 * @param synapse_id Synapse identifier
 * @param out_synapse Output: copy of synapse data (caller-allocated)
 * @return 0 on success, -1 if not found, -2 if NULL parameters
 */
int hetero_get_synapse_copy(hetero_system_t* system, uint32_t synapse_id,
                            hetero_synapse_t* out_synapse) {
    if (!system || !out_synapse) return -2;

    nimcp_platform_mutex_lock(system->mutex);

    int result = -1;
    for (size_t i = 0; i < system->num_synapses; i++) {
        if (system->synapses[i].synapse_id == synapse_id) {
            /* Copy synapse data while holding mutex */
            *out_synapse = system->synapses[i];
            result = 0;
            break;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);
    return result;
}

/* ============================================================================
 * Spatial Query API Implementation
 * ============================================================================ */

/* P2 fix: Add squared distance function to avoid expensive sqrtf in hot path
 * WHY:  sqrtf is ~10x slower than arithmetic. When comparing to threshold,
 *       d <= r is equivalent to d² <= r², avoiding the sqrt entirely.
 */
static inline float hetero_compute_distance_squared(
    const hetero_spatial_coords_t* pos1,
    const hetero_spatial_coords_t* pos2)
{
    if (!pos1 || !pos2) return 0.0f;

    float dx = pos1->x - pos2->x;
    float dy = pos1->y - pos2->y;
    float dz = pos1->z - pos2->z;

    return dx * dx + dy * dy + dz * dz;
}

float hetero_compute_distance(
    const hetero_spatial_coords_t* pos1,
    const hetero_spatial_coords_t* pos2)
{
    if (!pos1 || !pos2) return 0.0f;

    return sqrtf(hetero_compute_distance_squared(pos1, pos2));
}

float hetero_compute_depression_factor(
    float distance,
    float depression_factor,
    float decay_lambda)
{
    if (distance < 0.0f || decay_lambda <= 0.0f) return 0.0f;
    return depression_factor * expf(-distance / decay_lambda);
}

int hetero_find_neighbors(
    hetero_system_t* system,
    const hetero_spatial_coords_t* center_position,
    float radius,
    hetero_synapse_t** neighbors,
    size_t max_neighbors,
    size_t* num_found)
{
    if (!system || !center_position || !neighbors || !num_found) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    *num_found = 0;
    float radius_sq = radius * radius;

    /* Linear search through all synapses
     * P2 fix: Use squared distance comparison to avoid sqrtf in hot loop
     */
    for (size_t i = 0; i < system->num_synapses && *num_found < max_neighbors; i++) {
        hetero_synapse_t* syn = &system->synapses[i];
        float dist_sq = hetero_compute_distance_squared(center_position, &syn->position);

        if (dist_sq <= radius_sq && dist_sq > 0.0f) {  /* Exclude self (dist_sq=0) */
            neighbors[*num_found] = syn;
            (*num_found)++;
        }
    }

    return 0;
}

/* ============================================================================
 * Heterosynaptic Plasticity API Implementation
 * ============================================================================ */

int hetero_apply_depression(
    hetero_system_t* system,
    uint32_t potentiated_id,
    float ltp_amount,
    uint64_t current_time_ms)
{
    if (!system || ltp_amount <= 0.0f) return NIMCP_ERROR_INVALID_PARAM;

    hetero_synapse_t* potentiated = hetero_get_synapse(system, potentiated_id);
    if (!potentiated) return NIMCP_ERROR_INVALID_PARAM;

    /* Update potentiation tracking */
    potentiated->last_potentiation = ltp_amount;
    potentiated->last_potentiation_time_ms = current_time_ms;

    /* Find neighbors */
    hetero_synapse_t* neighbors[HETERO_MAX_NEIGHBORS_PER_SYNAPSE];
    size_t num_neighbors = 0;

    int result = hetero_find_neighbors(
        system,
        &potentiated->position,
        system->config.neighbor_radius,
        neighbors,
        HETERO_MAX_NEIGHBORS_PER_SYNAPSE,
        &num_neighbors
    );

    if (result != 0) return result;

    /* Apply depression to each neighbor */
    for (size_t i = 0; i < num_neighbors; i++) {
        hetero_synapse_t* neighbor = neighbors[i];

        /* Skip if same synapse */
        if (neighbor->synapse_id == potentiated_id) continue;

        /* Compute distance and depression factor */
        float distance = hetero_compute_distance(&potentiated->position, &neighbor->position);
        float factor = hetero_compute_depression_factor(
            distance,
            system->config.depression_factor,
            system->config.decay_lambda
        );

        /* Sleep modulation */
        if (system->config.enable_sleep_modulation) {
            factor = hetero_get_sleep_modulated_factor(system, factor);
        }

        /* Apply depression: Δw = -factor × LTP */
        float depression = -factor * ltp_amount;

        nimcp_platform_mutex_lock(neighbor->lock);
        /* Compute new weight and clamp atomically to avoid race window */
        float new_weight = neighbor->weight + depression;
        neighbor->weight = fmaxf(neighbor->w_min, fminf(neighbor->w_max, new_weight));
        neighbor->last_depression = -depression;  /* Store absolute value */
        neighbor->num_neighbor_depressions++;
        neighbor->total_hetero_ltd += -depression;
        nimcp_platform_mutex_unlock(neighbor->lock);
    }

    /* CRITICAL: Use atomic operations for thread-safe statistics update
     * WHY:  Statistics can be updated concurrently from multiple threads
     * HOW:  Use __atomic_fetch_add for counters, mutex for floats
     */
    __atomic_fetch_add(&system->total_depressions, num_neighbors, __ATOMIC_RELAXED);
    if (num_neighbors > 0) {
        /* Mutex needed for float EMA calculation (not atomically updateable) */
        nimcp_platform_mutex_lock(system->mutex);
        float alpha = NIMCP_EMA_WEIGHT_FAST;  /* Exponential moving average */
        system->avg_neighbors_per_competition =
            alpha * (float)num_neighbors + (NIMCP_EMA_WEIGHT_SLOW) * system->avg_neighbors_per_competition;
        nimcp_platform_mutex_unlock(system->mutex);
    }

    return 0;
}

int hetero_winner_take_all(
    hetero_system_t* system,
    const hetero_spatial_coords_t* center_position,
    float radius,
    hetero_competition_result_t* result)
{
    if (!system || !center_position || !result) return NIMCP_ERROR_NULL_POINTER;

    /* Use config radius if not specified */
    if (radius <= 0.0f) {
        radius = system->config.neighbor_radius;
    }

    /* Find all competitors */
    hetero_synapse_t* competitors[HETERO_MAX_NEIGHBORS_PER_SYNAPSE];
    size_t num_competitors = 0;

    int find_result = hetero_find_neighbors(
        system,
        center_position,
        radius,
        competitors,
        HETERO_MAX_NEIGHBORS_PER_SYNAPSE,
        &num_competitors
    );

    if (find_result != 0) return find_result;
    if (num_competitors == 0) {
        result->winner_id = 0;
        result->num_competitors = 0;
        return 0;
    }

    /* Find winner (strongest eligible synapse) */
    hetero_synapse_t* winner = NULL;
    float max_strength = system->config.wta_threshold;

    for (size_t i = 0; i < num_competitors; i++) {
        hetero_synapse_t* syn = competitors[i];
        if (syn->is_eligible_for_competition && syn->weight > max_strength) {
            max_strength = syn->weight;
            winner = syn;
        }
    }

    /* No winner found */
    if (!winner) {
        result->winner_id = 0;
        result->num_competitors = num_competitors;
        return 0;
    }

    /* Depress losers */
    result->depressed_ids = nimcp_malloc(num_competitors * sizeof(uint32_t));
    result->num_depressed = 0;
    float total_strength = 0.0f;

    for (size_t i = 0; i < num_competitors; i++) {
        hetero_synapse_t* syn = competitors[i];
        total_strength += syn->weight;

        if (syn->synapse_id != winner->synapse_id) {
            /* Apply suppression */
            nimcp_platform_mutex_lock(syn->lock);
            syn->weight *= system->config.wta_suppression_factor;
            syn->num_competitions++;
            nimcp_platform_mutex_unlock(syn->lock);

            result->depressed_ids[result->num_depressed++] = syn->synapse_id;
        }
    }

    /* Update winner */
    nimcp_platform_mutex_lock(winner->lock);
    winner->num_competitions++;
    winner->num_wins++;
    nimcp_platform_mutex_unlock(winner->lock);

    /* Fill result */
    result->winner_id = winner->synapse_id;
    result->num_competitors = num_competitors;
    result->winner_strength = winner->weight;
    result->avg_competitor_strength = num_competitors > 0 ? total_strength / num_competitors : 0.0f;

    /* CRITICAL: Use atomic increment for thread-safe statistics update
     * WHY:  total_competitions can be updated concurrently from multiple threads
     */
    __atomic_fetch_add(&system->total_competitions, 1, __ATOMIC_RELAXED);

    /* Callback */
    if (system->config.on_competition_event) {
        system->config.on_competition_event(
            result->winner_id,
            result->num_competitors,
            system->config.callback_user_data
        );
    }

    return 0;
}

float hetero_modulate_weight_change(
    hetero_system_t* system,
    uint32_t synapse_id,
    float homosynaptic_dw,
    uint64_t current_time_ms)
{
    if (!system) return homosynaptic_dw;

    hetero_synapse_t* syn = hetero_get_synapse(system, synapse_id);
    if (!syn) return homosynaptic_dw;

    /* Check for recent neighbor potentiation */
    float heterosynaptic_dw = 0.0f;

    /* This is simplified - in full implementation, would track neighbor events */
    /* For now, just return homosynaptic change */

    return homosynaptic_dw + heterosynaptic_dw;
}

/* ============================================================================
 * Sleep Integration API Implementation
 * ============================================================================ */

int hetero_set_sleep_state(hetero_system_t* system, sleep_state_t state) {
    if (!system) return NIMCP_ERROR_NULL_POINTER;

    nimcp_platform_mutex_lock(system->mutex);
    for (size_t i = 0; i < system->num_synapses; i++) {
        system->synapses[i].current_sleep_state = state;
    }
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

float hetero_get_sleep_modulated_factor(hetero_system_t* system, float base_factor) {
    if (!system || !system->config.enable_sleep_modulation) return base_factor;

    /* Use first synapse's sleep state as representative */
    if (system->num_synapses == 0) return base_factor;

    sleep_state_t state = system->synapses[0].current_sleep_state;

    switch (state) {
        case SLEEP_STATE_AWAKE:
            return base_factor * 1.0f;  /* Full competition */
        case SLEEP_STATE_DROWSY:
            return base_factor * 0.8f;
        case SLEEP_STATE_LIGHT_NREM:
            return base_factor * 0.5f;  /* Reduced for consolidation */
        case SLEEP_STATE_DEEP_NREM:
            return base_factor * 0.3f;  /* Minimal competition */
        case SLEEP_STATE_REM:
            return base_factor * 0.7f;  /* Moderate competition */
        default:
            return base_factor;
    }
}

/* ============================================================================
 * Bio-Async Integration API Implementation
 * ============================================================================ */

int hetero_connect_bio_async(hetero_system_t* system) {
    if (!system) return NIMCP_ERROR_NULL_POINTER;
    if (system->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = 0x0D40,  /* BIO_MODULE_IMMUNE_HETEROSYNAPTIC (would be added to nimcp_bio_messages.h) */
        .module_name = "heterosynaptic_plasticity",
        .inbox_capacity = NIMCP_INBOX_CAPACITY_SMALL,
        .user_data = system
    };

    system->bio_ctx = bio_router_register_module(&info);
    if (system->bio_ctx) {
        system->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected heterosynaptic to bio-async router");
        return 0;
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
        return NIMCP_ERROR_OPERATION_FAILED;
    }
}

int hetero_disconnect_bio_async(hetero_system_t* system) {
    if (!system || !system->bio_async_enabled) return 0;

    if (system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
    }

    system->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected heterosynaptic from bio-async router");
    return 0;
}

bool hetero_is_bio_async_connected(const hetero_system_t* system) {
    return system ? system->bio_async_enabled : false;
}

/* ============================================================================
 * Statistics API Implementation
 * ============================================================================ */

int hetero_get_statistics(
    const hetero_system_t* system,
    uint64_t* total_competitions,
    uint64_t* total_depressions,
    float* avg_neighbors)
{
    if (!system) return NIMCP_ERROR_NULL_POINTER;

    /* CRITICAL: Use atomic loads for thread-safe statistics access
     * WHY:  Statistics are updated outside lock in hot paths
     * HOW:  Use __atomic_load for consistent reads
     */
    if (total_competitions) {
        __atomic_load(&system->total_competitions, total_competitions, __ATOMIC_RELAXED);
    }
    if (total_depressions) {
        __atomic_load(&system->total_depressions, total_depressions, __ATOMIC_RELAXED);
    }
    if (avg_neighbors) {
        /* Float requires different atomic handling - use mutex for complex reads */
        nimcp_platform_mutex_lock(((hetero_system_t*)system)->mutex);
        *avg_neighbors = system->avg_neighbors_per_competition;
        nimcp_platform_mutex_unlock(((hetero_system_t*)system)->mutex);
    }

    return 0;
}

void hetero_reset_statistics(hetero_system_t* system) {
    if (!system) return;

    nimcp_platform_mutex_lock(system->mutex);
    system->total_competitions = 0;
    system->total_depressions = 0;
    system->avg_neighbors_per_competition = 0.0f;

    for (size_t i = 0; i < system->num_synapses; i++) {
        system->synapses[i].num_competitions = 0;
        system->synapses[i].num_wins = 0;
        system->synapses[i].num_neighbor_depressions = 0;
        system->synapses[i].total_hetero_ltd = 0.0f;
    }
    nimcp_platform_mutex_unlock(system->mutex);
}

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

void hetero_free_competition_result(hetero_competition_result_t* result) {
    if (!result) return;
    nimcp_free(result->depressed_ids);
    result->depressed_ids = NULL;
    result->num_depressed = 0;
}
