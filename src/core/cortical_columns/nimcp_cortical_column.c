#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_cortical_column.c - Cortical Column Implementation
//=============================================================================
/**
 * @file nimcp_cortical_column.c
 * @brief Implementation of cortical column architecture
 * @version 1.0.0
 * @date 2025-11-25
 *
 * Implementation follows NIMCP coding standards:
 * - WHAT/WHY/HOW documentation for all functions
 * - Guard clauses (early returns)
 * - Functions under 50 lines
 * - Memory pool allocation for hot paths
 * - Thread-safe with mutexes
 * - Comprehensive error checking
 */

#include "core/cortical_columns/nimcp_cortical_column.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

#define LOG_MODULE "cortical_column"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for cortical_column module */
static nimcp_health_agent_t* g_cortical_column_health_agent = NULL;

/**
 * @brief Set health agent for cortical_column heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void cortical_column_set_health_agent(nimcp_health_agent_t* agent) {
    g_cortical_column_health_agent = agent;
}

/** @brief Send heartbeat from cortical_column module */
static inline void cortical_column_heartbeat(const char* operation, float progress) {
    if (g_cortical_column_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_cortical_column_health_agent, operation, progress);
    }
}


// Logging macros
#define COLUMN_LOG_ERROR(...) LOG_ERROR(LOG_MODULE, __VA_ARGS__)
#define COLUMN_LOG_WARN(...) LOG_WARN(LOG_MODULE, __VA_ARGS__)
#define COLUMN_LOG_INFO(...) LOG_INFO(LOG_MODULE, __VA_ARGS__)
#define COLUMN_LOG_DEBUG(...) LOG_DEBUG(LOG_MODULE, __VA_ARGS__)

//=============================================================================
// Bio-Async Module Context (Thread-Safe Initialization)
//=============================================================================

static bio_module_context_t bio_ctx = NULL;
static bool bio_async_enabled = false;
static pthread_once_t bio_init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t bio_cleanup_mutex = PTHREAD_MUTEX_INITIALIZER;

static void cortical_column_bio_init_impl(void) {
    if (!bio_router_is_initialized()) {
        return;
    }

    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_CORTICAL_COLUMN,
        .module_name = "cortical_column",
        .inbox_capacity = 128,
        .user_data = NULL
    };

    bio_ctx = bio_router_register_module(&bio_info);
    if (bio_ctx) {
        bio_async_enabled = true;
        LOG_INFO(LOG_MODULE, "Bio-async registered for cortical_column module");
    }
}

__attribute__((constructor))
static void cortical_column_bio_init(void) {
    pthread_once(&bio_init_once, cortical_column_bio_init_impl);
}

__attribute__((destructor))
static void cortical_column_bio_cleanup(void) {
    pthread_mutex_lock(&bio_cleanup_mutex);
    if (bio_async_enabled && bio_ctx) {
        bio_router_unregister_module(bio_ctx);
        bio_ctx = NULL;
        bio_async_enabled = false;
        LOG_DEBUG(LOG_MODULE, "Bio-async unregistered for cortical_column module");
    }
    pthread_mutex_unlock(&bio_cleanup_mutex);
}

// Constants
#define DEFAULT_MAX_MINICOLUMNS 1000
#define DEFAULT_MAX_HYPERCOLUMNS 100
#define DEFAULT_MAX_NEURONS_PER_MINICOLUMN 100
#define DEFAULT_TEMPERATURE 1.0f
#define DEFAULT_LATERAL_INHIBITION_STRENGTH 0.5f
#define DEFAULT_LATERAL_INHIBITION_SIGMA1 1.0f
#define DEFAULT_LATERAL_INHIBITION_SIGMA2 3.0f
#define MIN_ACTIVATION 0.0f
#define MAX_ACTIVATION 1.0f
#define EPSILON 1e-7f

//=============================================================================
// Internal Data Structures
//=============================================================================

/**
 * @brief Minicolumn internal structure
 *
 * WHAT: Complete state for a single minicolumn
 * WHY:  Encapsulate all data needed for column computation
 */
struct minicolumn {
    // Configuration
    uint32_t* neuron_ids;              /**< Neuron indices (owned, must free) */
    uint32_t num_neurons;              /**< Number of neurons */
    receptive_field_t receptive_field; /**< Spatial receptive field */
    float tuning_preference;           /**< Feature preference */
    layer_distribution_t layers;       /**< Layer distribution */

    // Dynamic state
    float activation_level;            /**< Current activation [0.0, 1.0] */
    float inhibition_level;            /**< Current inhibition [0.0, 1.0] */
    float net_activation;              /**< Activation - inhibition */

    // Statistics
    uint32_t total_activations;        /**< Total activation count */
    float activation_sum;              /**< Sum for averaging */
    uint64_t last_activation_time_us;  /**< Last activation timestamp */

    // Memory management
    cortical_column_pool_t* pool;      /**< Parent pool */
    nimcp_platform_mutex_t mutex;      /**< Thread safety */
    bool initialized;                  /**< Initialization flag */
};

/**
 * @brief Hypercolumn internal structure
 *
 * WHAT: Complete state for a hypercolumn
 * WHY:  Manage collection of minicolumns with competition
 */
struct hypercolumn {
    // Configuration
    minicolumn_t** minicolumns;        /**< Array of minicolumn pointers */
    uint32_t num_minicolumns;          /**< Number of minicolumns */
    float feature_space_min;           /**< Min feature value */
    float feature_space_max;           /**< Max feature value */
    float topographic_x;               /**< X position in cortical map */
    float topographic_y;               /**< Y position in cortical map */

    // Competition parameters
    cc_competition_mode_t competition_mode; /**< Competition type */
    uint32_t k_winners;                /**< K for K-winners mode */
    float temperature;                 /**< Softmax temperature */

    // Lateral inhibition parameters
    float lateral_inhibition_strength; /**< Mexican hat amplitude */
    float lateral_inhibition_sigma1;   /**< Narrow excitation σ */
    float lateral_inhibition_sigma2;   /**< Wide inhibition σ */

    // Dynamic state
    float* activations;                /**< Cached activations array */
    uint32_t winner_index;             /**< Index of winner */
    float total_activation;            /**< Sum of activations */

    // Statistics
    uint32_t total_computations;       /**< Compute call count */
    float entropy;                     /**< Shannon entropy */

    // Memory management
    cortical_column_pool_t* pool;      /**< Parent pool */
    nimcp_platform_mutex_t mutex;      /**< Thread safety */
    bool initialized;                  /**< Initialization flag */
};

/**
 * @brief Memory pool structure
 *
 * WHAT: Pre-allocated memory for columns
 * WHY:  O(1) allocation, reduced fragmentation
 */
struct cortical_column_pool {
    // Configuration
    cortical_column_pool_config_t config;

    // Memory pools
    memory_pool_t minicolumn_pool;     /**< Pool for minicolumn structs */
    memory_pool_t hypercolumn_pool;    /**< Pool for hypercolumn structs */
    memory_pool_t neuron_id_pool;      /**< Pool for neuron ID arrays */
    memory_pool_t activation_pool;     /**< Pool for activation arrays */

    // Statistics
    uint32_t active_minicolumns;       /**< Currently active minicolumns */
    uint32_t active_hypercolumns;      /**< Currently active hypercolumns */
    uint32_t peak_minicolumns;         /**< Peak minicolumn usage */
    uint32_t peak_hypercolumns;        /**< Peak hypercolumn usage */

    // Thread safety
    nimcp_platform_mutex_t mutex;
    bool initialized;
};

//=============================================================================
// Forward Declarations
//=============================================================================

static float compute_gaussian_weight(float distance, float sigma);
static float compute_mexican_hat(float distance, float sigma1, float sigma2, float amplitude);
static void apply_softmax_inplace(float* activations, uint32_t size, float temperature);
static void apply_winner_take_all(float* activations, uint32_t size, uint32_t* winner_idx);
static void apply_k_winners(float* activations, uint32_t size, uint32_t k);
static float compute_entropy(const float* distribution, uint32_t size);
static float euclidean_distance_3d(float x1, float y1, float z1, float x2, float y2, float z2);

//=============================================================================
// Pool Management Implementation
//=============================================================================

/**
 * WHAT: Create cortical column memory pool
 * WHY:  Pre-allocate for O(1) allocation
 * HOW:  Create separate pools for each allocation type
 */
cortical_column_pool_t* cortical_column_pool_create(
    const cortical_column_pool_config_t* config
) {
    // Guard: validate input
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cortical_column_pool_create: config is NULL");
        COLUMN_LOG_ERROR("cortical_column_pool_create: NULL config");
        return NULL;
    }

    if (config->max_minicolumns == 0 || config->max_hypercolumns == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "cortical_column_pool_create: max_minicolumns or max_hypercolumns is zero");
        COLUMN_LOG_ERROR("cortical_column_pool_create: Invalid pool sizes");
        return NULL;
    }

    // Allocate pool structure
    cortical_column_pool_t* pool = nimcp_calloc(1, sizeof(cortical_column_pool_t));
    if (!pool) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to allocate pool");
        return NULL;
    }

    // Copy configuration
    pool->config = *config;

    // Initialize mutex
    if (nimcp_platform_mutex_init(&pool->mutex, false) != 0) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to initialize mutex");
        nimcp_free(pool);
        return NULL;
    }

    // Create minicolumn pool
    memory_pool_config_t mc_config = memory_pool_default_config(
        sizeof(minicolumn_t),
        config->max_minicolumns
    );
    pool->minicolumn_pool = memory_pool_create(&mc_config);
    if (!pool->minicolumn_pool) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to create minicolumn pool");
        nimcp_platform_mutex_destroy(&pool->mutex);
        nimcp_free(pool);
        return NULL;
    }

    // Create hypercolumn pool
    memory_pool_config_t hc_config = memory_pool_default_config(
        sizeof(hypercolumn_t),
        config->max_hypercolumns
    );
    pool->hypercolumn_pool = memory_pool_create(&hc_config);
    if (!pool->hypercolumn_pool) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to create hypercolumn pool");
        memory_pool_destroy(pool->minicolumn_pool);
        nimcp_platform_mutex_destroy(&pool->mutex);
        nimcp_free(pool);
        return NULL;
    }

    // Create neuron ID pool (for minicolumn neuron arrays)
    uint32_t max_neurons = config->max_neurons_per_minicolumn > 0
        ? config->max_neurons_per_minicolumn
        : DEFAULT_MAX_NEURONS_PER_MINICOLUMN;
    memory_pool_config_t nid_config = memory_pool_default_config(
        max_neurons * sizeof(uint32_t),
        config->max_minicolumns
    );
    pool->neuron_id_pool = memory_pool_create(&nid_config);
    if (!pool->neuron_id_pool) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to create neuron ID pool");
        memory_pool_destroy(pool->hypercolumn_pool);
        memory_pool_destroy(pool->minicolumn_pool);
        nimcp_platform_mutex_destroy(&pool->mutex);
        nimcp_free(pool);
        return NULL;
    }

    // Create activation array pool (for hypercolumn activations)
    memory_pool_config_t act_config = memory_pool_default_config(
        config->max_minicolumns * sizeof(float),
        config->max_hypercolumns
    );
    pool->activation_pool = memory_pool_create(&act_config);
    if (!pool->activation_pool) {
        COLUMN_LOG_ERROR("cortical_column_pool_create: Failed to create activation pool");
        memory_pool_destroy(pool->neuron_id_pool);
        memory_pool_destroy(pool->hypercolumn_pool);
        memory_pool_destroy(pool->minicolumn_pool);
        nimcp_platform_mutex_destroy(&pool->mutex);
        nimcp_free(pool);
        return NULL;
    }

    pool->initialized = true;
    COLUMN_LOG_INFO("Created cortical column pool: %u minicolumns, %u hypercolumns",
        config->max_minicolumns, config->max_hypercolumns);

    return pool;
}

/**
 * WHAT: Destroy cortical column pool
 * WHY:  Clean shutdown, free all memory
 * HOW:  Destroy all sub-pools and main structure
 */
void cortical_column_pool_destroy(cortical_column_pool_t* pool) {
    // Guard: validate input
    if (!pool) {
        return;
    }

    // Warn if columns still active
    if (pool->active_minicolumns > 0) {
        COLUMN_LOG_WARN("cortical_column_pool_destroy: %u minicolumns still active",
            pool->active_minicolumns);
    }
    if (pool->active_hypercolumns > 0) {
        COLUMN_LOG_WARN("cortical_column_pool_destroy: %u hypercolumns still active",
            pool->active_hypercolumns);
    }

    // Destroy pools
    if (pool->activation_pool) {
        memory_pool_destroy(pool->activation_pool);
    }
    if (pool->neuron_id_pool) {
        memory_pool_destroy(pool->neuron_id_pool);
    }
    if (pool->hypercolumn_pool) {
        memory_pool_destroy(pool->hypercolumn_pool);
    }
    if (pool->minicolumn_pool) {
        memory_pool_destroy(pool->minicolumn_pool);
    }

    // Destroy mutex
    nimcp_platform_mutex_destroy(&pool->mutex);

    // Free pool structure
    nimcp_free(pool);

    COLUMN_LOG_INFO("Destroyed cortical column pool");
}

//=============================================================================
// Minicolumn Lifecycle Implementation
//=============================================================================

/**
 * WHAT: Validate minicolumn configuration
 * WHY:  Catch errors early before allocation
 * HOW:  Check all required fields and constraints
 */
static bool validate_minicolumn_config(const minicolumn_config_t* config) {
    if (!config) {
        COLUMN_LOG_ERROR("validate_minicolumn_config: NULL config");
        return false;
    }

    if (!config->neuron_ids || config->num_neurons == 0) {
        COLUMN_LOG_ERROR("validate_minicolumn_config: Invalid neuron array");
        return false;
    }

    if (config->receptive_field.radius <= 0.0F) {
        COLUMN_LOG_ERROR("validate_minicolumn_config: Invalid receptive field radius");
        return false;
    }

    // Validate layer distribution
    uint32_t layer_sum = config->layers.layer_2_3_count +
                        config->layers.layer_4_count +
                        config->layers.layer_5_6_count;
    if (layer_sum != config->num_neurons) {
        COLUMN_LOG_ERROR("validate_minicolumn_config: Layer distribution sum (%u) != num_neurons (%u)",
            layer_sum, config->num_neurons);
        return false;
    }

    return true;
}

/**
 * WHAT: Create minicolumn
 * WHY:  Represent ~80-100 neurons with shared tuning
 * HOW:  Allocate from pool, copy config, initialize state
 */
minicolumn_t* minicolumn_create(
    cortical_column_pool_t* pool,
    const minicolumn_config_t* config
) {
    // Guard: validate inputs
    if (!pool) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_create: pool is NULL");
        COLUMN_LOG_ERROR("minicolumn_create: Invalid pool");
        return NULL;
    }
    if (!pool->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_create: pool not initialized");
        COLUMN_LOG_ERROR("minicolumn_create: Invalid pool");
        return NULL;
    }

    if (!validate_minicolumn_config(config)) {
        return NULL;
    }

    // Allocate minicolumn from pool
    minicolumn_t* col = (minicolumn_t*)memory_pool_acquire(pool->minicolumn_pool);
    if (!col) {
        COLUMN_LOG_ERROR("minicolumn_create: Pool exhausted");
        return NULL;
    }

    // Zero-initialize
    memset(col, 0, sizeof(minicolumn_t));

    // Allocate neuron ID array from pool
    col->neuron_ids = (uint32_t*)memory_pool_acquire(pool->neuron_id_pool);
    if (!col->neuron_ids) {
        COLUMN_LOG_ERROR("minicolumn_create: Failed to allocate neuron IDs");
        memory_pool_release(pool->minicolumn_pool, col);
        return NULL;
    }

    // Copy neuron IDs
    memcpy(col->neuron_ids, config->neuron_ids, config->num_neurons * sizeof(uint32_t));
    col->num_neurons = config->num_neurons;

    // Copy configuration
    col->receptive_field = config->receptive_field;
    col->tuning_preference = config->tuning_preference;
    col->layers = config->layers;

    // Initialize state
    col->activation_level = 0.0F;
    col->inhibition_level = 0.0F;
    col->net_activation = 0.0F;
    col->total_activations = 0;
    col->activation_sum = 0.0F;
    col->last_activation_time_us = 0;

    // Initialize mutex
    if (nimcp_platform_mutex_init(&col->mutex, false) != 0) {
        COLUMN_LOG_ERROR("minicolumn_create: Failed to initialize mutex");
        memory_pool_release(pool->neuron_id_pool, col->neuron_ids);
        memory_pool_release(pool->minicolumn_pool, col);
        return NULL;
    }

    col->pool = pool;
    col->initialized = true;

    // Update pool statistics
    nimcp_platform_mutex_lock(&pool->mutex);
    pool->active_minicolumns++;
    if (pool->active_minicolumns > pool->peak_minicolumns) {
        pool->peak_minicolumns = pool->active_minicolumns;
    }
    nimcp_platform_mutex_unlock(&pool->mutex);

    COLUMN_LOG_DEBUG("Created minicolumn: %u neurons, tuning=%.2f",
        col->num_neurons, col->tuning_preference);

    return col;
}

/**
 * WHAT: Destroy minicolumn
 * WHY:  Return resources to pool
 * HOW:  Release neuron IDs, destroy mutex, release struct
 */
void minicolumn_destroy(minicolumn_t* col) {
    // Guard: validate input
    if (!col || !col->initialized) {
        return;
    }

    cortical_column_pool_t* pool = col->pool;

    // Destroy mutex
    nimcp_platform_mutex_destroy(&col->mutex);

    // Release neuron IDs
    if (col->neuron_ids) {
        memory_pool_release(pool->neuron_id_pool, col->neuron_ids);
    }

    // Release minicolumn struct
    memory_pool_release(pool->minicolumn_pool, col);

    // Update pool statistics
    nimcp_platform_mutex_lock(&pool->mutex);
    if (pool->active_minicolumns > 0) {
        pool->active_minicolumns--;
    }
    nimcp_platform_mutex_unlock(&pool->mutex);

    COLUMN_LOG_DEBUG("Destroyed minicolumn");
}

//=============================================================================
// Hypercolumn Lifecycle Implementation
//=============================================================================

/**
 * WHAT: Validate hypercolumn configuration
 * WHY:  Catch errors early
 * HOW:  Check all fields and constraints
 */
static bool validate_hypercolumn_config(const hypercolumn_config_t* config) {
    if (!config) {
        COLUMN_LOG_ERROR("validate_hypercolumn_config: NULL config");
        return false;
    }

    if (config->num_minicolumns == 0) {
        COLUMN_LOG_ERROR("validate_hypercolumn_config: Zero minicolumns");
        return false;
    }

    if (!config->minicolumn_configs) {
        COLUMN_LOG_ERROR("validate_hypercolumn_config: NULL minicolumn configs");
        return false;
    }

    if (config->feature_space_min >= config->feature_space_max) {
        COLUMN_LOG_ERROR("validate_hypercolumn_config: Invalid feature space range");
        return false;
    }

    if (config->competition == CC_COMPETITION_K_WINNERS && config->k_winners == 0) {
        COLUMN_LOG_ERROR("validate_hypercolumn_config: K_WINNERS mode requires k_winners > 0");
        return false;
    }

    return true;
}

/**
 * WHAT: Create hypercolumn
 * WHY:  Organize minicolumns with competition
 * HOW:  Allocate from pool, create minicolumns, initialize competition
 */
hypercolumn_t* hypercolumn_create(
    cortical_column_pool_t* pool,
    const hypercolumn_config_t* config
) {
    // Guard: validate inputs
    if (!pool) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_create: pool is NULL");
        COLUMN_LOG_ERROR("hypercolumn_create: Invalid pool");
        return NULL;
    }
    if (!pool->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_create: pool not initialized");
        COLUMN_LOG_ERROR("hypercolumn_create: Invalid pool");
        return NULL;
    }

    if (!validate_hypercolumn_config(config)) {
        return NULL;
    }

    // Allocate hypercolumn from pool
    hypercolumn_t* hcol = (hypercolumn_t*)memory_pool_acquire(pool->hypercolumn_pool);
    if (!hcol) {
        COLUMN_LOG_ERROR("hypercolumn_create: Pool exhausted");
        return NULL;
    }

    // Zero-initialize
    memset(hcol, 0, sizeof(hypercolumn_t));

    // Allocate minicolumn pointer array
    hcol->minicolumns = nimcp_calloc(config->num_minicolumns, sizeof(minicolumn_t*));
    if (!hcol->minicolumns) {
        COLUMN_LOG_ERROR("hypercolumn_create: Failed to allocate minicolumn array");
        memory_pool_release(pool->hypercolumn_pool, hcol);
        return NULL;
    }

    // Allocate activation array from pool
    hcol->activations = (float*)memory_pool_acquire(pool->activation_pool);
    if (!hcol->activations) {
        COLUMN_LOG_ERROR("hypercolumn_create: Failed to allocate activation array");
        nimcp_free(hcol->minicolumns);
        memory_pool_release(pool->hypercolumn_pool, hcol);
        return NULL;
    }

    // Create all minicolumns
    for (uint32_t i = 0; i < config->num_minicolumns; i++) {
        hcol->minicolumns[i] = minicolumn_create(pool, &config->minicolumn_configs[i]);
        if (!hcol->minicolumns[i]) {
            COLUMN_LOG_ERROR("hypercolumn_create: Failed to create minicolumn %u", i);
            // Cleanup previously created minicolumns
            for (uint32_t j = 0; j < i; j++) {
                minicolumn_destroy(hcol->minicolumns[j]);
            }
            memory_pool_release(pool->activation_pool, hcol->activations);
            nimcp_free(hcol->minicolumns);
            memory_pool_release(pool->hypercolumn_pool, hcol);
            return NULL;
        }
    }

    // Copy configuration
    hcol->num_minicolumns = config->num_minicolumns;
    hcol->feature_space_min = config->feature_space_min;
    hcol->feature_space_max = config->feature_space_max;
    hcol->topographic_x = config->topographic_x;
    hcol->topographic_y = config->topographic_y;
    hcol->competition_mode = config->competition;
    hcol->k_winners = config->k_winners;
    hcol->temperature = config->temperature > 0.0F ? config->temperature : DEFAULT_TEMPERATURE;
    hcol->lateral_inhibition_strength = config->lateral_inhibition_strength > 0.0F
        ? config->lateral_inhibition_strength
        : DEFAULT_LATERAL_INHIBITION_STRENGTH;
    hcol->lateral_inhibition_sigma1 = config->lateral_inhibition_sigma1 > 0.0F
        ? config->lateral_inhibition_sigma1
        : DEFAULT_LATERAL_INHIBITION_SIGMA1;
    hcol->lateral_inhibition_sigma2 = config->lateral_inhibition_sigma2 > 0.0F
        ? config->lateral_inhibition_sigma2
        : DEFAULT_LATERAL_INHIBITION_SIGMA2;

    // Initialize state
    memset(hcol->activations, 0, config->num_minicolumns * sizeof(float));
    hcol->winner_index = 0;
    hcol->total_activation = 0.0F;
    hcol->total_computations = 0;
    hcol->entropy = 0.0F;

    // Initialize mutex
    if (nimcp_platform_mutex_init(&hcol->mutex, false) != 0) {
        COLUMN_LOG_ERROR("hypercolumn_create: Failed to initialize mutex");
        for (uint32_t i = 0; i < config->num_minicolumns; i++) {
            minicolumn_destroy(hcol->minicolumns[i]);
        }
        memory_pool_release(pool->activation_pool, hcol->activations);
        nimcp_free(hcol->minicolumns);
        memory_pool_release(pool->hypercolumn_pool, hcol);
        return NULL;
    }

    hcol->pool = pool;
    hcol->initialized = true;

    // Update pool statistics
    nimcp_platform_mutex_lock(&pool->mutex);
    pool->active_hypercolumns++;
    if (pool->active_hypercolumns > pool->peak_hypercolumns) {
        pool->peak_hypercolumns = pool->active_hypercolumns;
    }
    nimcp_platform_mutex_unlock(&pool->mutex);

    COLUMN_LOG_INFO("Created hypercolumn: %u minicolumns, feature range [%.2f, %.2f]",
        config->num_minicolumns, config->feature_space_min, config->feature_space_max);

    return hcol;
}

/**
 * WHAT: Destroy hypercolumn
 * WHY:  Free all resources
 * HOW:  Destroy minicolumns, release arrays, release struct
 */
void hypercolumn_destroy(hypercolumn_t* hcol) {
    // Guard: validate input
    if (!hcol || !hcol->initialized) {
        return;
    }

    cortical_column_pool_t* pool = hcol->pool;

    // Destroy all minicolumns
    if (hcol->minicolumns) {
        for (uint32_t i = 0; i < hcol->num_minicolumns; i++) {
            minicolumn_destroy(hcol->minicolumns[i]);
        }
        nimcp_free(hcol->minicolumns);
    }

    // Destroy mutex
    nimcp_platform_mutex_destroy(&hcol->mutex);

    // Release activation array
    if (hcol->activations) {
        memory_pool_release(pool->activation_pool, hcol->activations);
    }

    // Release hypercolumn struct
    memory_pool_release(pool->hypercolumn_pool, hcol);

    // Update pool statistics
    nimcp_platform_mutex_lock(&pool->mutex);
    if (pool->active_hypercolumns > 0) {
        pool->active_hypercolumns--;
    }
    nimcp_platform_mutex_unlock(&pool->mutex);

    COLUMN_LOG_DEBUG("Destroyed hypercolumn");
}

//=============================================================================
// Processing and Computation Implementation
//=============================================================================

/**
 * WHAT: Compute minicolumn activation
 * WHY:  Determine response to input
 * HOW:  Apply Gaussian receptive field weighting
 */
float minicolumn_compute(
    minicolumn_t* col,
    const float* input,
    uint32_t input_size
) {
    // Guard: validate inputs
    if (!col) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_compute: col is NULL");
        COLUMN_LOG_ERROR("minicolumn_compute: Invalid minicolumn");
        return -1.0F;
    }
    if (!col->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_compute: minicolumn not initialized");
        COLUMN_LOG_ERROR("minicolumn_compute: Invalid minicolumn");
        return -1.0F;
    }

    if (!input) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_compute: input is NULL");
        COLUMN_LOG_ERROR("minicolumn_compute: Invalid input");
        return -1.0F;
    }
    if (input_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "minicolumn_compute: input_size is zero");
        COLUMN_LOG_ERROR("minicolumn_compute: Invalid input");
        return -1.0F;
    }

    nimcp_platform_mutex_lock(&col->mutex);

    float activation;

    // Check if this is an orientation distribution input (>3 elements indicates distribution)
    // If tuning_preference is set and we have a distribution, use orientation-based computation
    if (input_size > 3 && col->tuning_preference >= 0.0F) {
        // Orientation-based computation:
        // Input is a distribution over orientations (0-180°)
        // Sample at the index corresponding to our tuning preference

        // Normalize tuning_preference to [0, 180) range
        float pref = col->tuning_preference;
        while (pref < 0.0F) pref += 180.0F;
        while (pref >= 180.0F) pref -= 180.0F;

        // Compute the fractional index into the input distribution
        float frac_idx = pref / 180.0F * (float)(input_size - 1);
        uint32_t idx_low = (uint32_t)frac_idx;
        uint32_t idx_high = idx_low + 1;
        float t = frac_idx - (float)idx_low;

        // Handle wraparound for orientation
        if (idx_high >= input_size) {
            idx_high = 0;
        }

        // Linear interpolation to sample the distribution at our tuning preference
        activation = (1.0F - t) * input[idx_low] + t * input[idx_high];
    } else {
        // Spatial receptive field computation (original logic)
        // Compute input centroid (simplified - assumes input is 3D feature vector)
        float input_x = input_size > 0 ? input[0] : 0.0F;
        float input_y = input_size > 1 ? input[1] : 0.0F;
        float input_z = input_size > 2 ? input[2] : 0.0F;

        // Compute distance from receptive field center
        float distance = euclidean_distance_3d(
            input_x, input_y, input_z,
            col->receptive_field.center_x,
            col->receptive_field.center_y,
            col->receptive_field.center_z
        );

        // Apply Gaussian weighting: w(d) = exp(-d²/2σ²)
        activation = compute_gaussian_weight(distance, col->receptive_field.radius);
    }

    // Apply inhibition
    float net_activation = fmaxf(0.0F, activation - col->inhibition_level);

    // Update state
    col->activation_level = net_activation;
    col->net_activation = net_activation;
    col->total_activations++;
    col->activation_sum += net_activation;

    float result = col->activation_level;

    nimcp_platform_mutex_unlock(&col->mutex);

    return result;
}

/**
 * WHAT: Compute lateral inhibition values
 * WHY:  Implement Mexican hat competition
 * HOW:  Compute inhibition for each minicolumn based on neighbors
 */
static void compute_lateral_inhibition_internal(hypercolumn_t* hcol) {
    // For each minicolumn, compute inhibition from all others
    for (uint32_t i = 0; i < hcol->num_minicolumns; i++) {
        minicolumn_t* col_i = hcol->minicolumns[i];
        float total_inhibition = 0.0F;

        for (uint32_t j = 0; j < hcol->num_minicolumns; j++) {
            if (i == j) continue;

            minicolumn_t* col_j = hcol->minicolumns[j];

            // Compute distance in feature space
            float feature_distance = fabsf(col_i->tuning_preference - col_j->tuning_preference);

            // Mexican hat: excitation - inhibition
            float lateral_effect = compute_mexican_hat(
                feature_distance,
                hcol->lateral_inhibition_sigma1,
                hcol->lateral_inhibition_sigma2,
                hcol->lateral_inhibition_strength
            );

            // Multiply by neighbor activation
            total_inhibition += lateral_effect * hcol->activations[j];
        }

        // Apply inhibition (clamp to positive)
        col_i->inhibition_level = fmaxf(0.0F, total_inhibition);
    }
}

/**
 * WHAT: Compute hypercolumn response
 * WHY:  Process all minicolumns and run competition
 * HOW:  Compute activations, apply lateral inhibition, run competition
 */
void hypercolumn_compute(
    hypercolumn_t* hcol,
    const float* input,
    uint32_t input_size
) {
    // Process pending bio-async messages
    if (bio_ctx) {
        bio_router_process_inbox(bio_ctx, 5);
    }

    // Guard: validate inputs
    if (!hcol) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_compute: hcol is NULL");
        COLUMN_LOG_ERROR("hypercolumn_compute: Invalid hypercolumn");
        return;
    }
    if (!hcol->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_compute: hypercolumn not initialized");
        COLUMN_LOG_ERROR("hypercolumn_compute: Invalid hypercolumn");
        return;
    }

    if (!input) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_compute: input is NULL");
        COLUMN_LOG_ERROR("hypercolumn_compute: Invalid input");
        return;
    }
    if (input_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "hypercolumn_compute: input_size is zero");
        COLUMN_LOG_ERROR("hypercolumn_compute: Invalid input");
        return;
    }

    nimcp_platform_mutex_lock(&hcol->mutex);

    // Step 1: Compute activation for each minicolumn
    for (uint32_t i = 0; i < hcol->num_minicolumns; i++) {
        float activation = minicolumn_compute(hcol->minicolumns[i], input, input_size);
        hcol->activations[i] = fmaxf(0.0F, activation);
    }

    // Step 2: Apply lateral inhibition (Mexican hat)
    compute_lateral_inhibition_internal(hcol);

    // Re-compute activations after inhibition
    for (uint32_t i = 0; i < hcol->num_minicolumns; i++) {
        minicolumn_t* col = hcol->minicolumns[i];
        hcol->activations[i] = fmaxf(0.0F, col->activation_level - col->inhibition_level);
    }

    // Step 3: Run competition
    hypercolumn_run_competition(hcol, hcol->competition_mode, hcol->temperature);

    // Step 4: Update statistics
    hcol->total_computations++;

    // Compute total activation
    float total = 0.0F;
    for (uint32_t i = 0; i < hcol->num_minicolumns; i++) {
        total += hcol->activations[i];
    }
    hcol->total_activation = total;

    // Compute entropy
    hcol->entropy = compute_entropy(hcol->activations, hcol->num_minicolumns);

    nimcp_platform_mutex_unlock(&hcol->mutex);
}

/**
 * WHAT: Get winning minicolumn index
 * WHY:  Determine detected feature
 * HOW:  Return cached winner from last compute
 */
uint32_t hypercolumn_get_winner(hypercolumn_t* hcol) {
    // Guard: validate input
    if (!hcol) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_get_winner: hcol is NULL");
        COLUMN_LOG_ERROR("hypercolumn_get_winner: Invalid hypercolumn");
        return UINT32_MAX;
    }
    if (!hcol->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_get_winner: hypercolumn not initialized");
        COLUMN_LOG_ERROR("hypercolumn_get_winner: Invalid hypercolumn");
        return UINT32_MAX;
    }

    nimcp_platform_mutex_lock(&hcol->mutex);
    uint32_t winner = hcol->winner_index;
    nimcp_platform_mutex_unlock(&hcol->mutex);

    return winner;
}

/**
 * WHAT: Get activation distribution
 * WHY:  Access full population code
 * HOW:  Copy activation array to output
 */
void hypercolumn_get_distribution(
    hypercolumn_t* hcol,
    float* out_distribution,
    uint32_t size
) {
    // Guard: validate inputs
    if (!hcol) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_get_distribution: hcol is NULL");
        COLUMN_LOG_ERROR("hypercolumn_get_distribution: Invalid hypercolumn");
        return;
    }
    if (!hcol->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_get_distribution: hypercolumn not initialized");
        COLUMN_LOG_ERROR("hypercolumn_get_distribution: Invalid hypercolumn");
        return;
    }

    if (!out_distribution) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_get_distribution: out_distribution is NULL");
        COLUMN_LOG_ERROR("hypercolumn_get_distribution: NULL output");
        return;
    }

    if (size < hcol->num_minicolumns) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "hypercolumn_get_distribution: size too small");
        COLUMN_LOG_ERROR("hypercolumn_get_distribution: Output too small (%u < %u)",
            size, hcol->num_minicolumns);
        return;
    }

    nimcp_platform_mutex_lock(&hcol->mutex);
    memcpy(out_distribution, hcol->activations, hcol->num_minicolumns * sizeof(float));
    nimcp_platform_mutex_unlock(&hcol->mutex);
}

//=============================================================================
// Lateral Inhibition and Competition Implementation
//=============================================================================

/**
 * WHAT: Apply lateral inhibition to minicolumn
 * WHY:  Reduce activation based on competition
 * HOW:  Subtract inhibition, clamp to [0, 1]
 */
void minicolumn_apply_lateral_inhibition(minicolumn_t* col, float inhibition) {
    // Guard: validate inputs
    if (!col) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_apply_lateral_inhibition: col is NULL");
        COLUMN_LOG_ERROR("minicolumn_apply_lateral_inhibition: Invalid minicolumn");
        return;
    }
    if (!col->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_apply_lateral_inhibition: minicolumn not initialized");
        COLUMN_LOG_ERROR("minicolumn_apply_lateral_inhibition: Invalid minicolumn");
        return;
    }

    nimcp_platform_mutex_lock(&col->mutex);

    col->inhibition_level = fmaxf(0.0F, fminf(1.0F, inhibition));
    col->net_activation = fmaxf(0.0F, col->activation_level - col->inhibition_level);

    nimcp_platform_mutex_unlock(&col->mutex);
}

/**
 * WHAT: Run competition in hypercolumn
 * WHY:  Implement winner-take-all or softmax dynamics
 * HOW:  Apply selected competition algorithm
 */
void hypercolumn_run_competition(
    hypercolumn_t* hcol,
    cc_competition_mode_t mode,
    float temperature
) {
    // Guard: validate input
    if (!hcol) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_run_competition: hcol is NULL");
        COLUMN_LOG_ERROR("hypercolumn_run_competition: Invalid hypercolumn");
        return;
    }
    if (!hcol->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_run_competition: hypercolumn not initialized");
        COLUMN_LOG_ERROR("hypercolumn_run_competition: Invalid hypercolumn");
        return;
    }

    // Already locked by hypercolumn_compute in typical usage
    // Lock here for standalone calls
    bool need_lock = true;
    if (nimcp_platform_mutex_trylock(&hcol->mutex) != 0) {
        need_lock = false;  // Already locked by caller
    }

    switch (mode) {
        case CC_COMPETITION_WINNER_TAKE_ALL:
            apply_winner_take_all(hcol->activations, hcol->num_minicolumns, &hcol->winner_index);
            break;

        case CC_COMPETITION_K_WINNERS:
            apply_k_winners(hcol->activations, hcol->num_minicolumns, hcol->k_winners);
            // Find winner (highest activation)
            hcol->winner_index = 0;
            for (uint32_t i = 1; i < hcol->num_minicolumns; i++) {
                if (hcol->activations[i] > hcol->activations[hcol->winner_index]) {
                    hcol->winner_index = i;
                }
            }
            break;

        case CC_COMPETITION_SOFTMAX:
            apply_softmax_inplace(hcol->activations, hcol->num_minicolumns, temperature);
            // Find winner (highest probability)
            hcol->winner_index = 0;
            for (uint32_t i = 1; i < hcol->num_minicolumns; i++) {
                if (hcol->activations[i] > hcol->activations[hcol->winner_index]) {
                    hcol->winner_index = i;
                }
            }
            break;

        case CC_COMPETITION_NONE:
            // No competition - find max activation
            hcol->winner_index = 0;
            for (uint32_t i = 1; i < hcol->num_minicolumns; i++) {
                if (hcol->activations[i] > hcol->activations[hcol->winner_index]) {
                    hcol->winner_index = i;
                }
            }
            break;
    }

    if (need_lock) {
        nimcp_platform_mutex_unlock(&hcol->mutex);
    }
}

//=============================================================================
// Receptive Field Operations Implementation
//=============================================================================

/**
 * WHAT: Set minicolumn receptive field
 * WHY:  Configure or adapt receptive field
 * HOW:  Update internal structure
 */
void minicolumn_set_receptive_field(
    minicolumn_t* col,
    float cx, float cy, float cz,
    float radius
) {
    // Guard: validate inputs
    if (!col) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_set_receptive_field: col is NULL");
        COLUMN_LOG_ERROR("minicolumn_set_receptive_field: Invalid minicolumn");
        return;
    }
    if (!col->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_set_receptive_field: minicolumn not initialized");
        COLUMN_LOG_ERROR("minicolumn_set_receptive_field: Invalid minicolumn");
        return;
    }

    if (radius <= 0.0F) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "minicolumn_set_receptive_field: radius must be positive");
        COLUMN_LOG_ERROR("minicolumn_set_receptive_field: Invalid radius");
        return;
    }

    nimcp_platform_mutex_lock(&col->mutex);

    col->receptive_field.center_x = cx;
    col->receptive_field.center_y = cy;
    col->receptive_field.center_z = cz;
    col->receptive_field.radius = radius;

    nimcp_platform_mutex_unlock(&col->mutex);
}

/**
 * WHAT: Compute receptive field weight
 * WHY:  Determine point activation strength
 * HOW:  Apply Gaussian: w(d) = exp(-d²/2σ²)
 */
float minicolumn_compute_receptive_weight(
    minicolumn_t* col,
    float x, float y, float z
) {
    // Guard: validate input
    if (!col) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_compute_receptive_weight: col is NULL");
        COLUMN_LOG_ERROR("minicolumn_compute_receptive_weight: Invalid minicolumn");
        return -1.0F;
    }
    if (!col->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_compute_receptive_weight: minicolumn not initialized");
        COLUMN_LOG_ERROR("minicolumn_compute_receptive_weight: Invalid minicolumn");
        return -1.0F;
    }

    nimcp_platform_mutex_lock(&col->mutex);

    float distance = euclidean_distance_3d(
        x, y, z,
        col->receptive_field.center_x,
        col->receptive_field.center_y,
        col->receptive_field.center_z
    );

    float weight = compute_gaussian_weight(distance, col->receptive_field.radius);

    nimcp_platform_mutex_unlock(&col->mutex);

    return weight;
}

//=============================================================================
// Statistics and Monitoring Implementation
//=============================================================================

/**
 * WHAT: Get minicolumn statistics
 * WHY:  Monitor activity and performance
 * HOW:  Copy internal stats to output
 */
void minicolumn_get_stats(minicolumn_t* col, minicolumn_stats_t* stats) {
    // Guard: validate inputs
    if (!col) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_get_stats: col is NULL");
        COLUMN_LOG_ERROR("minicolumn_get_stats: Invalid minicolumn");
        return;
    }
    if (!col->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "minicolumn_get_stats: minicolumn not initialized");
        COLUMN_LOG_ERROR("minicolumn_get_stats: Invalid minicolumn");
        return;
    }

    if (!stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "minicolumn_get_stats: stats is NULL");
        COLUMN_LOG_ERROR("minicolumn_get_stats: NULL stats");
        return;
    }

    nimcp_platform_mutex_lock(&col->mutex);

    stats->activation_level = col->activation_level;
    stats->inhibition_level = col->inhibition_level;
    stats->total_activations = col->total_activations;
    stats->average_activation = col->total_activations > 0
        ? col->activation_sum / col->total_activations
        : 0.0F;
    stats->tuning_preference = col->tuning_preference;
    stats->num_neurons = col->num_neurons;
    stats->last_activation_time_us = col->last_activation_time_us;

    nimcp_platform_mutex_unlock(&col->mutex);
}

/**
 * WHAT: Get hypercolumn statistics
 * WHY:  Monitor competition dynamics
 * HOW:  Copy internal stats to output
 */
void hypercolumn_get_stats(hypercolumn_t* hcol, cc_hypercolumn_stats_t* stats) {
    // Guard: validate inputs
    if (!hcol) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_get_stats: hcol is NULL");
        COLUMN_LOG_ERROR("hypercolumn_get_stats: Invalid hypercolumn");
        return;
    }
    if (!hcol->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE, "hypercolumn_get_stats: hypercolumn not initialized");
        COLUMN_LOG_ERROR("hypercolumn_get_stats: Invalid hypercolumn");
        return;
    }

    if (!stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "hypercolumn_get_stats: stats is NULL");
        COLUMN_LOG_ERROR("hypercolumn_get_stats: NULL stats");
        return;
    }

    nimcp_platform_mutex_lock(&hcol->mutex);

    stats->num_minicolumns = hcol->num_minicolumns;
    stats->winner_index = hcol->winner_index;
    stats->winner_activation = hcol->winner_index < hcol->num_minicolumns
        ? hcol->activations[hcol->winner_index]
        : 0.0F;
    stats->total_activation = hcol->total_activation;
    stats->entropy = hcol->entropy;
    stats->total_computations = hcol->total_computations;
    stats->competition_mode = hcol->competition_mode;

    nimcp_platform_mutex_unlock(&hcol->mutex);
}

//=============================================================================
// Mathematical Helper Functions
//=============================================================================

/**
 * WHAT: Compute Gaussian weight
 * WHY:  Receptive field weighting
 * HOW:  w(d) = exp(-d²/2σ²)
 */
static float compute_gaussian_weight(float distance, float sigma) {
    float sigma_sq = sigma * sigma;
    return expf(-(distance * distance) / (2.0F * sigma_sq));
}

/**
 * WHAT: Compute Mexican hat function
 * WHY:  Lateral inhibition (difference of Gaussians)
 * HOW:  I(d) = (1 - d²/σ₁²)exp(-d²/2σ₁²) - A*exp(-d²/2σ₂²)
 */
static float compute_mexican_hat(float distance, float sigma1, float sigma2, float amplitude) {
    float d_sq = distance * distance;
    float sigma1_sq = sigma1 * sigma1;
    float sigma2_sq = sigma2 * sigma2;

    float excitation = (1.0F - d_sq / sigma1_sq) * expf(-d_sq / (2.0F * sigma1_sq));
    float inhibition = amplitude * expf(-d_sq / (2.0F * sigma2_sq));

    return excitation - inhibition;
}

/**
 * WHAT: Apply softmax in-place
 * WHY:  Convert activations to probability distribution
 * HOW:  p_i = exp(a_i/T) / Σexp(a_j/T)
 */
static void apply_softmax_inplace(float* activations, uint32_t size, float temperature) {
    if (!activations || size == 0 || temperature <= 0.0F) {
        return;
    }

    // Find max for numerical stability
    float max_val = activations[0];
    for (uint32_t i = 1; i < size; i++) {
        if (activations[i] > max_val) {
            max_val = activations[i];
        }
    }

    // Compute exp(a_i/T - max/T) and sum
    float sum = 0.0F;
    for (uint32_t i = 0; i < size; i++) {
        activations[i] = expf((activations[i] - max_val) / temperature);
        sum += activations[i];
    }

    // Normalize
    if (sum > EPSILON) {
        for (uint32_t i = 0; i < size; i++) {
            activations[i] /= sum;
        }
    }
}

/**
 * WHAT: Apply winner-take-all
 * WHY:  Binary competition - only winner active
 * HOW:  Set max=1.0, others=0.0
 */
static void apply_winner_take_all(float* activations, uint32_t size, uint32_t* winner_idx) {
    if (!activations || size == 0 || !winner_idx) {
        return;
    }

    // Find winner
    uint32_t max_idx = 0;
    float max_val = activations[0];
    for (uint32_t i = 1; i < size; i++) {
        if (activations[i] > max_val) {
            max_val = activations[i];
            max_idx = i;
        }
    }

    // Set winner=1.0, others=0.0
    for (uint32_t i = 0; i < size; i++) {
        activations[i] = (i == max_idx) ? 1.0F : 0.0F;
    }

    *winner_idx = max_idx;
}

/**
 * WHAT: Apply K-winners competition
 * WHY:  Sparse distributed representation
 * HOW:  Set top K=1.0, others=0.0
 */
static void apply_k_winners(float* activations, uint32_t size, uint32_t k) {
    if (!activations || size == 0 || k == 0 || k > size) {
        return;
    }

    // Create index array for sorting
    uint32_t* indices = nimcp_malloc(size * sizeof(uint32_t));
    if (!indices) {
        return;
    }

    for (uint32_t i = 0; i < size; i++) {
        indices[i] = i;
    }

    // Partial selection sort for top K
    for (uint32_t i = 0; i < k; i++) {
        uint32_t max_idx = i;
        for (uint32_t j = i + 1; j < size; j++) {
            if (activations[indices[j]] > activations[indices[max_idx]]) {
                max_idx = j;
            }
        }
        // Swap
        uint32_t temp = indices[i];
        indices[i] = indices[max_idx];
        indices[max_idx] = temp;
    }

    // Set top K=1.0, others=0.0
    for (uint32_t i = 0; i < size; i++) {
        bool is_winner = false;
        for (uint32_t j = 0; j < k; j++) {
            if (i == indices[j]) {
                is_winner = true;
                break;
            }
        }
        activations[i] = is_winner ? 1.0F : 0.0F;
    }

    nimcp_free(indices);
}

/**
 * WHAT: Compute Shannon entropy
 * WHY:  Measure distribution uncertainty
 * HOW:  H = -Σ p_i log₂(p_i)
 */
static float compute_entropy(const float* distribution, uint32_t size) {
    if (!distribution || size == 0) {
        return 0.0F;
    }

    // Compute sum for normalization
    float sum = 0.0F;
    for (uint32_t i = 0; i < size; i++) {
        sum += distribution[i];
    }

    if (sum < EPSILON) {
        return 0.0F;
    }

    // Compute entropy: H = -Σ p_i log₂(p_i)
    float entropy = 0.0F;
    for (uint32_t i = 0; i < size; i++) {
        float p = distribution[i] / sum;
        if (p > EPSILON) {
            entropy -= p * log2f(p);
        }
    }

    return entropy;
}

/**
 * WHAT: Compute Euclidean distance in 3D
 * WHY:  Distance for receptive field calculations
 * HOW:  d = √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)
 */
static float euclidean_distance_3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

/**
 * WHAT: Query knowledge graph for cortical column module self-knowledge
 * WHY:  Enable self-awareness and introspection about this module's role
 * HOW:  Query KG for entity info, log observations, check relations
 */
int cortical_column_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Cortical_Column_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            COLUMN_LOG_DEBUG("Cortical column self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Cortical_Column_Module");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Cortical_Column_Module");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
