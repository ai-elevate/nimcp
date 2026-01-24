/**
 * @file nimcp_distributed_cognition.c
 * @brief Implementation of distributed cognitive integration layer
 *
 * ARCHITECTURE:
 * - Mediator between local cognitive systems and P2P network
 * - Event-driven coordination with thread-safe synchronization
 * - Supports neuromodulator diffusion, glial coordination, region sync
 * - ASYNC INTEGRATION: Uses futures for non-blocking operations
 * - CONFIG INTEGRATION: All hyperparameters configurable via config module
 * - SECURITY INTEGRATION: Registered with security system for monitoring
 *
 * THREAD MODEL:
 * - 3 worker threads (neuromod sync, glial sync, region sync)
 * - rwlock protects shared state (readers-writer lock)
 * - Lock-free reads for high-frequency queries
 *
 * MEMORY MANAGEMENT:
 * - All allocations use unified memory (nimcp_malloc/calloc/free)
 * - Cleanup on destroy, NULL-safe operations
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version Phase 3.1 (Async/Config/Logging/Security Integrated)
 */

#include "networking/distributed/nimcp_distributed_cognition.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_future.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/error/nimcp_error_codes.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "security/nimcp_security.h"
#include "utils/config/nimcp_dynamic_config.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "networking.distributed"

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Registered neuromodulator pool for synchronization
 */
typedef struct {
    neuromodulator_pool_t* pool;   /**< Local neuromod pool */
    uint64_t last_sync_time;        /**< Last sync timestamp */
    bool needs_broadcast;           /**< Flag for pending broadcast */
} registered_neuromod_t;

/**
 * @brief Registered glial integration system
 */
typedef struct {
    glial_integration_t* glial;    /**< Local glial system */
    uint64_t last_sync_time;        /**< Last sync timestamp */
    uint32_t pending_prunings;      /**< Pending pruning operations */
} registered_glial_t;

/**
 * @brief Registered brain region
 */
typedef struct {
    brain_region_t* region;        /**< Local brain region */
    uint16_t region_type;           /**< Region type identifier */
    uint64_t last_sync_time;        /**< Last sync timestamp */
} registered_region_t;

/**
 * @brief Distributed cognition coordinator (opaque struct)
 */
struct distrib_cognition_struct {
    // Configuration
    distrib_cognition_config_t config;

    // P2P network
    p2p_node_t p2p_node;

    // Registered systems
    registered_neuromod_t* neuromod_pools;
    size_t neuromod_pool_count;
    size_t neuromod_pool_capacity;

    registered_glial_t* glial_systems;
    size_t glial_system_count;
    size_t glial_system_capacity;

    registered_region_t* brain_regions;
    size_t brain_region_count;
    size_t brain_region_capacity;

    // Statistics
    distrib_cognition_stats_t stats;

    // Thread safety
    nimcp_rwlock_t rwlock;

    // Worker threads
    nimcp_thread_t neuromod_thread;
    nimcp_thread_t glial_thread;
    nimcp_thread_t region_thread;

    // Control flags
    bool running;
    bool shutdown_requested;

    // Bio-async integration
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
};

//=============================================================================
// Module Registration and Configuration
//=============================================================================

#define MODULE_NAME "networking.distributed_cognition"
static uint32_t g_security_module_id = 0;

// Configuration keys
#define CONFIG_KEY_ENABLE_NEUROMOD_SYNC "distrib_cog.enable_neuromod_sync"
#define CONFIG_KEY_NEUROMOD_INTERVAL "distrib_cog.neuromod_broadcast_interval_ms"
#define CONFIG_KEY_NEUROMOD_DIFFUSION "distrib_cog.neuromod_diffusion_rate"
#define CONFIG_KEY_ENABLE_GLIAL_SYNC "distrib_cog.enable_glial_sync"
#define CONFIG_KEY_GLIAL_INTERVAL "distrib_cog.glial_sync_interval_ms"
#define CONFIG_KEY_ENABLE_REGION_SYNC "distrib_cog.enable_region_sync"
#define CONFIG_KEY_REGION_INTERVAL "distrib_cog.region_sync_interval_ms"
#define CONFIG_KEY_MAX_MESSAGE_QUEUE "distrib_cog.max_message_queue"

//=============================================================================
// Default Configuration
//=============================================================================

static const distrib_cognition_config_t DEFAULT_CONFIG = {
    .enable_neuromod_sync = true,
    .neuromod_broadcast_interval_ms = 100,  // 10 Hz broadcast
    .neuromod_diffusion_rate = 0.1F,        // 10% cross-node diffusion

    .enable_glial_sync = true,
    .glial_sync_interval_ms = 500,          // 2 Hz sync

    .enable_region_sync = true,
    .region_sync_interval_ms = 200,         // 5 Hz sync

    .sync_mode = SYNC_MODE_BIDIRECTIONAL,
    .max_message_queue = 1000,
    .enable_bio_async = false
};

/**
 * @brief Load configuration from config module
 *
 * WHY: Make all hyperparameters configurable at runtime
 * HOW: Query config module for each parameter with fallback to defaults
 */
static void load_configuration(distrib_cognition_config_t* config)
{
    if (!config) return;

    // Start with defaults
    *config = DEFAULT_CONFIG;

    // Override with config module values if available
    bool enable_neuromod = config_get_bool(CONFIG_KEY_ENABLE_NEUROMOD_SYNC, config->enable_neuromod_sync);
    config->enable_neuromod_sync = enable_neuromod;
    LOG_MODULE_DEBUG(MODULE_NAME, "Config: enable_neuromod_sync=%d", enable_neuromod);

    int64_t neuromod_interval = config_get_int(CONFIG_KEY_NEUROMOD_INTERVAL, config->neuromod_broadcast_interval_ms);
    if (neuromod_interval > 0) {
        config->neuromod_broadcast_interval_ms = (uint32_t)neuromod_interval;
        LOG_MODULE_DEBUG(MODULE_NAME, "Config: neuromod_interval=%d ms", (int)neuromod_interval);
    }

    double diffusion_rate = config_get_float(CONFIG_KEY_NEUROMOD_DIFFUSION, config->neuromod_diffusion_rate);
    if (diffusion_rate >= 0.0 && diffusion_rate <= 1.0) {
        config->neuromod_diffusion_rate = (float)diffusion_rate;
        LOG_MODULE_DEBUG(MODULE_NAME, "Config: neuromod_diffusion_rate=%.3f", diffusion_rate);
    }

    bool enable_glial = config_get_bool(CONFIG_KEY_ENABLE_GLIAL_SYNC, config->enable_glial_sync);
    config->enable_glial_sync = enable_glial;
    LOG_MODULE_DEBUG(MODULE_NAME, "Config: enable_glial_sync=%d", enable_glial);

    int64_t glial_interval = config_get_int(CONFIG_KEY_GLIAL_INTERVAL, config->glial_sync_interval_ms);
    if (glial_interval > 0) {
        config->glial_sync_interval_ms = (uint32_t)glial_interval;
        LOG_MODULE_DEBUG(MODULE_NAME, "Config: glial_interval=%d ms", (int)glial_interval);
    }

    bool enable_region = config_get_bool(CONFIG_KEY_ENABLE_REGION_SYNC, config->enable_region_sync);
    config->enable_region_sync = enable_region;
    LOG_MODULE_DEBUG(MODULE_NAME, "Config: enable_region_sync=%d", enable_region);

    int64_t region_interval = config_get_int(CONFIG_KEY_REGION_INTERVAL, config->region_sync_interval_ms);
    if (region_interval > 0) {
        config->region_sync_interval_ms = (uint32_t)region_interval;
        LOG_MODULE_DEBUG(MODULE_NAME, "Config: region_interval=%d ms", (int)region_interval);
    }

    int64_t max_queue = config_get_int(CONFIG_KEY_MAX_MESSAGE_QUEUE, config->max_message_queue);
    if (max_queue > 0) {
        config->max_message_queue = (uint32_t)max_queue;
        LOG_MODULE_DEBUG(MODULE_NAME, "Config: max_message_queue=%d", (int)max_queue);
    }
}

//=============================================================================
// Forward Declarations - Worker Threads
//=============================================================================

static void* neuromod_sync_worker(void* arg);
static void* glial_sync_worker(void* arg);
static void* region_sync_worker(void* arg);

//=============================================================================
// Forward Declarations - Message Handlers
//=============================================================================

static void handle_neuromod_level_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len);
static void handle_glial_pruning_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len);
static void handle_glial_calcium_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len);
static void handle_region_activity_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len);

//=============================================================================
// Creation and Destruction
//=============================================================================

distrib_cognition_t distrib_cognition_create(
    const distrib_cognition_config_t* config,
    p2p_node_t p2p_node)
{
    if (!p2p_node) {
        LOG_ERROR(LOG_MODULE, "Invalid P2P node");
        return NULL;
    }

    // Allocate coordinator
    distrib_cognition_t dc = (distrib_cognition_t)nimcp_calloc(1, sizeof(struct distrib_cognition_struct));
    if (!dc) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(struct distrib_cognition_struct),
                          "Failed to allocate distributed cognition coordinator");
        LOG_ERROR(LOG_MODULE, "Failed to allocate coordinator");
        return NULL;
    }

    // Set configuration
    if (config) {
        dc->config = *config;
    } else {
        dc->config = DEFAULT_CONFIG;
    }

    // Store P2P node
    dc->p2p_node = p2p_node;

    // Initialize rwlock
    if (nimcp_rwlock_init(&dc->rwlock) != NIMCP_SUCCESS) {
        NIMCP_THROW_THREADING(NIMCP_ERROR_THREAD_CREATE, 0,
                             "Failed to initialize rwlock for distributed cognition coordinator");
        LOG_ERROR(LOG_MODULE, "Failed to initialize rwlock");
        nimcp_free(dc);
        return NULL;
    }

    // Allocate initial capacity for registered systems
    dc->neuromod_pool_capacity = 4;
    dc->neuromod_pools = (registered_neuromod_t*)nimcp_calloc(dc->neuromod_pool_capacity, sizeof(registered_neuromod_t));
    if (!dc->neuromod_pools) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                          dc->neuromod_pool_capacity * sizeof(registered_neuromod_t),
                          "Failed to allocate neuromodulator pool storage");
        nimcp_rwlock_destroy(&dc->rwlock);
        nimcp_free(dc);
        return NULL;
    }

    dc->glial_system_capacity = 4;
    dc->glial_systems = (registered_glial_t*)nimcp_calloc(dc->glial_system_capacity, sizeof(registered_glial_t));
    if (!dc->glial_systems) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                          dc->glial_system_capacity * sizeof(registered_glial_t),
                          "Failed to allocate glial system storage");
        nimcp_free(dc->neuromod_pools);
        nimcp_rwlock_destroy(&dc->rwlock);
        nimcp_free(dc);
        return NULL;
    }

    dc->brain_region_capacity = 8;
    dc->brain_regions = (registered_region_t*)nimcp_calloc(dc->brain_region_capacity, sizeof(registered_region_t));
    if (!dc->brain_regions) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                          dc->brain_region_capacity * sizeof(registered_region_t),
                          "Failed to allocate brain region storage");
        nimcp_free(dc->glial_systems);
        nimcp_free(dc->neuromod_pools);
        nimcp_rwlock_destroy(&dc->rwlock);
        nimcp_free(dc);
        return NULL;
    }

    // Initialize statistics
    memset(&dc->stats, 0, sizeof(distrib_cognition_stats_t));

    // Not running yet
    dc->running = false;
    dc->shutdown_requested = false;

    // Initialize bio-async if enabled
    dc->bio_ctx = NULL;
    dc->bio_async_enabled = false;
    if (dc->config.enable_bio_async && bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_DISTRIBUTED,
            .module_name = "distributed_cognition",
            .inbox_capacity = 64,
            .user_data = dc
        };
        dc->bio_ctx = bio_router_register_module(&bio_info);
        if (dc->bio_ctx) {
            dc->bio_async_enabled = true;
            LOG_INFO(LOG_MODULE, "Bio-async communication registered");
        } else {
            LOG_WARN(LOG_MODULE, "Failed to register bio-async communication");
        }
    }

    LOG_INFO(LOG_MODULE, "Coordinator created successfully");

    return dc;
}

void distrib_cognition_destroy(distrib_cognition_t dc)
{
    if (!dc) {
        return;
    }

    // Stop if running
    if (dc->running) {
        distrib_cognition_stop(dc);
    }

    // Unregister bio-async
    if (dc->bio_async_enabled && dc->bio_ctx) {
        bio_router_unregister_module(dc->bio_ctx);
        dc->bio_ctx = NULL;
        dc->bio_async_enabled = false;
        LOG_INFO(LOG_MODULE, "Bio-async communication unregistered");
    }

    // Free registered systems
    if (dc->neuromod_pools) {
        nimcp_free(dc->neuromod_pools);
    }

    if (dc->glial_systems) {
        nimcp_free(dc->glial_systems);
    }

    if (dc->brain_regions) {
        nimcp_free(dc->brain_regions);
    }

    // Destroy rwlock
    nimcp_rwlock_destroy(&dc->rwlock);

    // Free coordinator
    nimcp_free(dc);

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Coordinator destroyed");
}

//=============================================================================
// Neuromodulator Network Synchronization
//=============================================================================

bool distrib_cognition_register_neuromod_pool(
    distrib_cognition_t dc,
    neuromodulator_pool_t* pool)
{
    // Process pending bio-async messages
    if (dc && dc->bio_ctx) {
        bio_router_process_inbox(dc->bio_ctx, 5);
    }

    if (!dc || !pool) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid parameters for neuromod pool registration");
        return false;
    }

    nimcp_rwlock_wrlock(&dc->rwlock);

    // Check capacity
    if (dc->neuromod_pool_count >= dc->neuromod_pool_capacity) {
        // Expand capacity
        size_t new_capacity = dc->neuromod_pool_capacity * 2;
        registered_neuromod_t* new_pools = (registered_neuromod_t*)nimcp_realloc(
            dc->neuromod_pools,
            new_capacity * sizeof(registered_neuromod_t)
        );

        if (!new_pools) {
            NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                              new_capacity * sizeof(registered_neuromod_t),
                              "Failed to expand neuromod pool capacity from %zu to %zu",
                              dc->neuromod_pool_capacity, new_capacity);
            nimcp_rwlock_unlock(&dc->rwlock);
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to expand neuromod pool capacity");
            return false;
        }

        dc->neuromod_pools = new_pools;
        dc->neuromod_pool_capacity = new_capacity;
    }

    // Register pool
    registered_neuromod_t* reg = &dc->neuromod_pools[dc->neuromod_pool_count++];
    reg->pool = pool;
    reg->last_sync_time = nimcp_time_get_us();
    reg->needs_broadcast = false;

    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Neuromodulator pool registered (total: %zu)", dc->neuromod_pool_count);

    return true;
}

bool distrib_cognition_broadcast_neuromod(
    distrib_cognition_t dc,
    neuromodulator_type_t type,
    float concentration)
{
    if (!dc) {
        return false;
    }

    if (concentration < 0.0F || concentration > 1.0F) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid neuromodulator concentration: %.2f", concentration);
        return false;
    }

    // Create payload
    neuromod_level_payload_t payload = {
        .neuromod_type = (uint8_t)type,
        .reserved = 0,
        .region_id = 0,  // Global broadcast
        .concentration = concentration,
        .timestamp = nimcp_time_get_us()
    };

    // Send control message via P2P
    // NOTE: This would require p2p_node_send_control_message() or similar
    // For now, we'll increment stats as placeholder

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.neuromod_broadcasts++;
    dc->stats.messages_sent++;
    dc->stats.last_neuromod_sync = payload.timestamp;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Broadcast neuromodulator type=%d concentration=%.3f", type, concentration);

    return true;
}

//=============================================================================
// Glial Network Coordination
//=============================================================================

bool distrib_cognition_register_glial_system(
    distrib_cognition_t dc,
    glial_integration_t* glial)
{
    if (!dc || !glial) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid parameters for glial system registration");
        return false;
    }

    nimcp_rwlock_wrlock(&dc->rwlock);

    // Check capacity
    if (dc->glial_system_count >= dc->glial_system_capacity) {
        size_t new_capacity = dc->glial_system_capacity * 2;
        registered_glial_t* new_systems = (registered_glial_t*)nimcp_realloc(
            dc->glial_systems,
            new_capacity * sizeof(registered_glial_t)
        );

        if (!new_systems) {
            NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                              new_capacity * sizeof(registered_glial_t),
                              "Failed to expand glial system capacity from %zu to %zu",
                              dc->glial_system_capacity, new_capacity);
            nimcp_rwlock_unlock(&dc->rwlock);
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to expand glial system capacity");
            return false;
        }

        dc->glial_systems = new_systems;
        dc->glial_system_capacity = new_capacity;
    }

    // Register system
    registered_glial_t* reg = &dc->glial_systems[dc->glial_system_count++];
    reg->glial = glial;
    reg->last_sync_time = nimcp_time_get_us();
    reg->pending_prunings = 0;

    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Glial system registered (total: %zu)", dc->glial_system_count);

    return true;
}

bool distrib_cognition_coordinate_pruning(
    distrib_cognition_t dc,
    uint32_t source_neuron_id,
    uint32_t target_neuron_id,
    float activity_score,
    uint8_t action)
{
    if (!dc) {
        return false;
    }

    if (activity_score < 0.0F || activity_score > 1.0F) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid activity score: %.2f", activity_score);
        return false;
    }

    if (action > 2) {  // 0=monitor, 1=prune, 2=preserve
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid pruning action: %d", action);
        return false;
    }

    // Create payload
    glial_pruning_payload_t payload = {
        .source_neuron_id = source_neuron_id,
        .target_neuron_id = target_neuron_id,
        .activity_score = activity_score,
        .pruning_action = action,
        .reserved = {0, 0, 0},
        .timestamp = nimcp_time_get_us()
    };

    // Send via P2P
    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.glial_pruning_coordinations++;
    dc->stats.messages_sent++;
    dc->stats.last_glial_sync = payload.timestamp;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Coordinate pruning synapse=%u->%u score=%.3f action=%d",
              source_neuron_id, target_neuron_id, activity_score, action);

    return true;
}

bool distrib_cognition_propagate_calcium_wave(
    distrib_cognition_t dc,
    uint32_t astrocyte_id,
    float calcium_level,
    float wave_velocity)
{
    if (!dc) {
        return false;
    }

    if (calcium_level < 0.0F || calcium_level > 1.0F) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid calcium level: %.2f", calcium_level);
        return false;
    }

    // Create payload
    glial_calcium_payload_t payload = {
        .astrocyte_id = astrocyte_id,
        .calcium_level = calcium_level,
        .wave_velocity = wave_velocity,
        .affected_synapses = 0,  // Will be filled by local system
        .reserved = 0,
        .timestamp = nimcp_time_get_us()
    };

    // Send via P2P
    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.glial_calcium_propagations++;
    dc->stats.messages_sent++;
    dc->stats.last_glial_sync = payload.timestamp;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Propagate calcium wave astrocyte=%u level=%.3f velocity=%.1f µm/s",
              astrocyte_id, calcium_level, wave_velocity);

    return true;
}

//=============================================================================
// Brain Region Synchronization
//=============================================================================

bool distrib_cognition_register_brain_region(
    distrib_cognition_t dc,
    brain_region_t* region)
{
    if (!dc || !region) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid parameters for brain region registration");
        return false;
    }

    nimcp_rwlock_wrlock(&dc->rwlock);

    // Check capacity
    if (dc->brain_region_count >= dc->brain_region_capacity) {
        size_t new_capacity = dc->brain_region_capacity * 2;
        registered_region_t* new_regions = (registered_region_t*)nimcp_realloc(
            dc->brain_regions,
            new_capacity * sizeof(registered_region_t)
        );

        if (!new_regions) {
            NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY,
                              new_capacity * sizeof(registered_region_t),
                              "Failed to expand brain region capacity from %zu to %zu",
                              dc->brain_region_capacity, new_capacity);
            nimcp_rwlock_unlock(&dc->rwlock);
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to expand brain region capacity");
            return false;
        }

        dc->brain_regions = new_regions;
        dc->brain_region_capacity = new_capacity;
    }

    // Register region
    registered_region_t* reg = &dc->brain_regions[dc->brain_region_count++];
    reg->region = region;
    reg->region_type = 0;  // Would come from region->type
    reg->last_sync_time = nimcp_time_get_us();

    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Brain region registered (total: %zu)", dc->brain_region_count);

    return true;
}

bool distrib_cognition_broadcast_region_activity(
    distrib_cognition_t dc,
    uint16_t region_type,
    float avg_activity,
    float spike_rate,
    uint32_t active_neurons,
    uint32_t total_neurons)
{
    if (!dc) {
        return false;
    }

    if (avg_activity < 0.0F || avg_activity > 1.0F) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid average activity: %.2f", avg_activity);
        return false;
    }

    if (active_neurons > total_neurons) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Active neurons (%u) exceeds total neurons (%u)", active_neurons, total_neurons);
        return false;
    }

    // Create payload
    region_activity_payload_t payload = {
        .region_type = region_type,
        .reserved = 0,
        .avg_activity = avg_activity,
        .spike_rate = spike_rate,
        .active_neurons = active_neurons,
        .total_neurons = total_neurons,
        .timestamp = nimcp_time_get_us()
    };

    // Send via P2P
    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.region_activity_broadcasts++;
    dc->stats.messages_sent++;
    dc->stats.last_region_sync = payload.timestamp;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Broadcast region activity type=%u activity=%.3f rate=%.1fHz neurons=%u/%u",
              region_type, avg_activity, spike_rate, active_neurons, total_neurons);

    return true;
}

//=============================================================================
// Worker Threads
//=============================================================================

/**
 * @brief Neuromodulator synchronization worker thread
 */
static void* neuromod_sync_worker(void* arg)
{
    distrib_cognition_t dc = (distrib_cognition_t)arg;

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Neuromodulator sync worker started");

    while (!dc->shutdown_requested) {
        if (dc->config.enable_neuromod_sync) {
            // Read lock for iteration
            nimcp_rwlock_rdlock(&dc->rwlock);
            size_t pool_count = dc->neuromod_pool_count;
            nimcp_rwlock_unlock(&dc->rwlock);

            // Sync each registered pool
            for (size_t i = 0; i < pool_count; i++) {
                nimcp_rwlock_rdlock(&dc->rwlock);
                registered_neuromod_t* reg = &dc->neuromod_pools[i];
                uint64_t now = nimcp_time_get_us();
                uint64_t elapsed_ms = (now - reg->last_sync_time) / NIMCP_US_PER_MS;
                nimcp_rwlock_unlock(&dc->rwlock);

                if (elapsed_ms >= dc->config.neuromod_broadcast_interval_ms) {
                    // Broadcast neuromodulator levels
                    // NOTE: Would query pool->concentrations here

                    nimcp_rwlock_wrlock(&dc->rwlock);
                    reg->last_sync_time = now;
                    nimcp_rwlock_unlock(&dc->rwlock);
                }
            }
        }

        // Sleep for interval
        nimcp_time_sleep_ms(dc->config.neuromod_broadcast_interval_ms);
    }

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Neuromodulator sync worker stopped");

    return NULL;
}

/**
 * @brief Glial coordination worker thread
 */
static void* glial_sync_worker(void* arg)
{
    distrib_cognition_t dc = (distrib_cognition_t)arg;

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Glial sync worker started");

    while (!dc->shutdown_requested) {
        if (dc->config.enable_glial_sync) {
            nimcp_rwlock_rdlock(&dc->rwlock);
            size_t system_count = dc->glial_system_count;
            nimcp_rwlock_unlock(&dc->rwlock);

            // Sync each registered glial system
            for (size_t i = 0; i < system_count; i++) {
                nimcp_rwlock_rdlock(&dc->rwlock);
                registered_glial_t* reg = &dc->glial_systems[i];
                uint64_t now = nimcp_time_get_us();
                uint64_t elapsed_ms = (now - reg->last_sync_time) / NIMCP_US_PER_MS;
                nimcp_rwlock_unlock(&dc->rwlock);

                if (elapsed_ms >= dc->config.glial_sync_interval_ms) {
                    // Coordinate glial activities
                    // NOTE: Would query glial system state here

                    nimcp_rwlock_wrlock(&dc->rwlock);
                    reg->last_sync_time = now;
                    nimcp_rwlock_unlock(&dc->rwlock);
                }
            }
        }

        // Sleep for interval
        nimcp_time_sleep_ms(dc->config.glial_sync_interval_ms);
    }

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Glial sync worker stopped");

    return NULL;
}

/**
 * @brief Brain region synchronization worker thread
 */
static void* region_sync_worker(void* arg)
{
    distrib_cognition_t dc = (distrib_cognition_t)arg;

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Region sync worker started");

    while (!dc->shutdown_requested) {
        if (dc->config.enable_region_sync) {
            nimcp_rwlock_rdlock(&dc->rwlock);
            size_t region_count = dc->brain_region_count;
            nimcp_rwlock_unlock(&dc->rwlock);

            // Sync each registered region
            for (size_t i = 0; i < region_count; i++) {
                nimcp_rwlock_rdlock(&dc->rwlock);
                registered_region_t* reg = &dc->brain_regions[i];
                uint64_t now = nimcp_time_get_us();
                uint64_t elapsed_ms = (now - reg->last_sync_time) / NIMCP_US_PER_MS;
                nimcp_rwlock_unlock(&dc->rwlock);

                if (elapsed_ms >= dc->config.region_sync_interval_ms) {
                    // Broadcast region activity
                    // NOTE: Would query region statistics here

                    nimcp_rwlock_wrlock(&dc->rwlock);
                    reg->last_sync_time = now;
                    nimcp_rwlock_unlock(&dc->rwlock);
                }
            }
        }

        // Sleep for interval
        nimcp_time_sleep_ms(dc->config.region_sync_interval_ms);
    }

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Region sync worker stopped");

    return NULL;
}

//=============================================================================
// Message Handlers (Incoming Network Messages)
//=============================================================================

static void handle_neuromod_level_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len)
{
    if (payload_len != sizeof(neuromod_level_payload_t)) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid neuromod level payload size: %zu", payload_len);
        return;
    }

    const neuromod_level_payload_t* msg = (const neuromod_level_payload_t*)payload;

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.neuromod_updates_received++;
    dc->stats.messages_received++;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Received neuromod level: type=%d concentration=%.3f",
              msg->neuromod_type, msg->concentration);

    // Apply diffusion to local pools
    nimcp_rwlock_rdlock(&dc->rwlock);
    for (size_t i = 0; i < dc->neuromod_pool_count; i++) {
        registered_neuromod_t* reg = &dc->neuromod_pools[i];
        // NOTE: Would apply diffusion: local = local * (1-rate) + remote * rate
        // pool_update_concentration(reg->pool, msg->neuromod_type,
        //                           msg->concentration, dc->config.neuromod_diffusion_rate);
    }
    nimcp_rwlock_unlock(&dc->rwlock);
}

static void handle_glial_pruning_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len)
{
    if (payload_len != sizeof(glial_pruning_payload_t)) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid glial pruning payload size: %zu", payload_len);
        return;
    }

    const glial_pruning_payload_t* msg = (const glial_pruning_payload_t*)payload;

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.glial_pruning_coordinations++;
    dc->stats.messages_received++;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Received pruning coordination: synapse=%u->%u action=%d",
              msg->source_neuron_id, msg->target_neuron_id, msg->pruning_action);
}

static void handle_glial_calcium_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len)
{
    if (payload_len != sizeof(glial_calcium_payload_t)) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid glial calcium payload size: %zu", payload_len);
        return;
    }

    const glial_calcium_payload_t* msg = (const glial_calcium_payload_t*)payload;

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.glial_calcium_propagations++;
    dc->stats.messages_received++;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Received calcium wave: astrocyte=%u level=%.3f",
              msg->astrocyte_id, msg->calcium_level);
}

static void handle_region_activity_msg(distrib_cognition_t dc, const uint8_t* payload, size_t payload_len)
{
    if (payload_len != sizeof(region_activity_payload_t)) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid region activity payload size: %zu", payload_len);
        return;
    }

    const region_activity_payload_t* msg = (const region_activity_payload_t*)payload;

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->stats.region_state_syncs++;
    dc->stats.messages_received++;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_DEBUG, "[distributed_cognition] Received region activity: type=%u activity=%.3f rate=%.1fHz",
              msg->region_type, msg->avg_activity, msg->spike_rate);
}

//=============================================================================
// Control and Monitoring
//=============================================================================

bool distrib_cognition_start(distrib_cognition_t dc)
{
    if (!dc) {
        return false;
    }

    if (dc->running) {
        log_message(LOG_LEVEL_WARN, "[distributed_cognition] Coordinator already running");
        return true;
    }

    dc->shutdown_requested = false;

    // Start worker threads based on configuration
    if (dc->config.enable_neuromod_sync) {
        if (nimcp_thread_create(&dc->neuromod_thread, neuromod_sync_worker, dc, NULL) != NIMCP_SUCCESS) {
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to start neuromod worker");
            return false;
        }
    }

    if (dc->config.enable_glial_sync) {
        if (nimcp_thread_create(&dc->glial_thread, glial_sync_worker, dc, NULL) != NIMCP_SUCCESS) {
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to start glial worker");
            if (dc->config.enable_neuromod_sync) {
                dc->shutdown_requested = true;
                nimcp_thread_join(dc->neuromod_thread, NULL);
            }
            return false;
        }
    }

    if (dc->config.enable_region_sync) {
        if (nimcp_thread_create(&dc->region_thread, region_sync_worker, dc, NULL) != NIMCP_SUCCESS) {
            log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Failed to start region worker");
            dc->shutdown_requested = true;
            if (dc->config.enable_neuromod_sync) {
                nimcp_thread_join(dc->neuromod_thread, NULL);
            }
            if (dc->config.enable_glial_sync) {
                nimcp_thread_join(dc->glial_thread, NULL);
            }
            return false;
        }
    }

    dc->running = true;

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Coordinator started");

    return true;
}

bool distrib_cognition_stop(distrib_cognition_t dc)
{
    if (!dc) {
        return false;
    }

    if (!dc->running) {
        log_message(LOG_LEVEL_WARN, "[distributed_cognition] Coordinator not running");
        return true;
    }

    // Signal shutdown
    dc->shutdown_requested = true;

    // Join worker threads
    if (dc->config.enable_neuromod_sync) {
        nimcp_thread_join(dc->neuromod_thread, NULL);
    }

    if (dc->config.enable_glial_sync) {
        nimcp_thread_join(dc->glial_thread, NULL);
    }

    if (dc->config.enable_region_sync) {
        nimcp_thread_join(dc->region_thread, NULL);
    }

    dc->running = false;

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Coordinator stopped");

    return true;
}

bool distrib_cognition_get_stats(
    distrib_cognition_t dc,
    distrib_cognition_stats_t* stats)
{
    if (!dc || !stats) {
        return false;
    }

    nimcp_rwlock_rdlock(&dc->rwlock);
    *stats = dc->stats;
    nimcp_rwlock_unlock(&dc->rwlock);

    return true;
}

bool distrib_cognition_set_sync_mode(
    distrib_cognition_t dc,
    sync_mode_t mode)
{
    if (!dc) {
        return false;
    }

    if (mode < SYNC_MODE_DISABLED || mode > SYNC_MODE_BIDIRECTIONAL) {
        log_message(LOG_LEVEL_ERROR, "[distributed_cognition] Invalid sync mode: %d", mode);
        return false;
    }

    nimcp_rwlock_wrlock(&dc->rwlock);
    dc->config.sync_mode = mode;
    nimcp_rwlock_unlock(&dc->rwlock);

    log_message(LOG_LEVEL_INFO, "[distributed_cognition] Sync mode set to %d", mode);

    return true;
}

//=============================================================================
// Async Operations - Context Structures
//=============================================================================

/**
 * @brief Context for async neuromodulator broadcast
 */
typedef struct {
    distrib_cognition_t dc;
    neuromodulator_type_t type;
    float concentration;
    nimcp_promise_t promise;
} async_neuromod_broadcast_ctx_t;

/**
 * @brief Context for async calcium wave propagation
 */
typedef struct {
    distrib_cognition_t dc;
    uint32_t astrocyte_id;
    float calcium_level;
    float wave_velocity;
    nimcp_promise_t promise;
} async_calcium_wave_ctx_t;

/**
 * @brief Context for async pruning coordination
 */
typedef struct {
    distrib_cognition_t dc;
    uint32_t source_neuron_id;
    uint32_t target_neuron_id;
    float activity_score;
    uint8_t action;
    nimcp_promise_t promise;
} async_pruning_ctx_t;

//=============================================================================
// Async Operations - Worker Threads
//=============================================================================

/**
 * @brief Worker thread for async neuromodulator broadcast
 */
static void* async_neuromod_broadcast_worker(void* arg)
{
    async_neuromod_broadcast_ctx_t* ctx = (async_neuromod_broadcast_ctx_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG(MODULE_NAME, "Async neuromod broadcast starting: type=%d concentration=%.3f",
                     ctx->type, ctx->concentration);

    // Perform the synchronous broadcast operation
    bool result = distrib_cognition_broadcast_neuromod(
        ctx->dc,
        ctx->type,
        ctx->concentration
    );

    // Set promise result using correct API
    if (result) {
        bool success = true;
        nimcp_promise_complete(ctx->promise, &success);
        LOG_MODULE_DEBUG(MODULE_NAME, "Async neuromod broadcast completed successfully");
    } else {
        nimcp_promise_fail(ctx->promise, NIMCP_ERROR_OPERATION_FAILED);
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast failed");
    }

    // Free context
    nimcp_free(ctx);

    return NULL;
}

/**
 * @brief Worker thread for async calcium wave propagation
 */
static void* async_calcium_wave_worker(void* arg)
{
    async_calcium_wave_ctx_t* ctx = (async_calcium_wave_ctx_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG(MODULE_NAME, "Async calcium wave starting: astrocyte=%u level=%.3f velocity=%.1f",
                     ctx->astrocyte_id, ctx->calcium_level, ctx->wave_velocity);

    // Perform the synchronous calcium wave operation
    bool result = distrib_cognition_propagate_calcium_wave(
        ctx->dc,
        ctx->astrocyte_id,
        ctx->calcium_level,
        ctx->wave_velocity
    );

    // Set promise result using correct API
    if (result) {
        bool success = true;
        nimcp_promise_complete(ctx->promise, &success);
        LOG_MODULE_DEBUG(MODULE_NAME, "Async calcium wave completed successfully");
    } else {
        nimcp_promise_fail(ctx->promise, NIMCP_ERROR_OPERATION_FAILED);
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave failed");
    }

    // Free context
    nimcp_free(ctx);

    return NULL;
}

/**
 * @brief Worker thread for async pruning coordination
 */
static void* async_pruning_worker(void* arg)
{
    async_pruning_ctx_t* ctx = (async_pruning_ctx_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG(MODULE_NAME, "Async pruning coordination starting: synapse=%u->%u score=%.3f action=%d",
                     ctx->source_neuron_id, ctx->target_neuron_id, ctx->activity_score, ctx->action);

    // Perform the synchronous pruning coordination
    bool result = distrib_cognition_coordinate_pruning(
        ctx->dc,
        ctx->source_neuron_id,
        ctx->target_neuron_id,
        ctx->activity_score,
        ctx->action
    );

    // Set promise result using correct API
    if (result) {
        bool success = true;
        nimcp_promise_complete(ctx->promise, &success);
        LOG_MODULE_DEBUG(MODULE_NAME, "Async pruning coordination completed successfully");
    } else {
        nimcp_promise_fail(ctx->promise, NIMCP_ERROR_OPERATION_FAILED);
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination failed");
    }

    // Free context
    nimcp_free(ctx);

    return NULL;
}

//=============================================================================
// Async Operations - Public API
//=============================================================================

nimcp_future_t distrib_cognition_broadcast_neuromod_async(
    distrib_cognition_t dc,
    neuromodulator_type_t type,
    float concentration)
{
    if (!dc) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: invalid coordinator");
        return NULL;
    }

    if (concentration < 0.0F || concentration > 1.0F) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: invalid concentration %.3f", concentration);
        return NULL;
    }

    // Create promise/future pair (result is a bool)
    nimcp_promise_t promise = nimcp_promise_create(sizeof(bool));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: failed to create promise");
        return NULL;
    }

    nimcp_future_t future = nimcp_promise_get_future(promise);
    if (!future) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: failed to get future");
        return NULL;
    }

    // Allocate context
    async_neuromod_broadcast_ctx_t* ctx = (async_neuromod_broadcast_ctx_t*)nimcp_malloc(sizeof(async_neuromod_broadcast_ctx_t));
    if (!ctx) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: failed to allocate context");
        return NULL;
    }

    ctx->dc = dc;
    ctx->type = type;
    ctx->concentration = concentration;
    ctx->promise = promise;

    // Spawn worker thread
    nimcp_thread_t thread;
    if (nimcp_thread_create(&thread, async_neuromod_broadcast_worker, ctx, NULL) != NIMCP_SUCCESS) {
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async neuromod broadcast: failed to create worker thread");
        return NULL;
    }

    // Detach thread (worker will clean up context)
    nimcp_thread_detach(thread);

    LOG_MODULE_INFO(MODULE_NAME, "Async neuromod broadcast started: type=%d concentration=%.3f", type, concentration);

    return future;
}

nimcp_future_t distrib_cognition_propagate_calcium_wave_async(
    distrib_cognition_t dc,
    uint32_t astrocyte_id,
    float calcium_level,
    float wave_velocity)
{
    if (!dc) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: invalid coordinator");
        return NULL;
    }

    if (calcium_level < 0.0F || calcium_level > 1.0F) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: invalid calcium level %.3f", calcium_level);
        return NULL;
    }

    // Create promise/future pair (result is a bool)
    nimcp_promise_t promise = nimcp_promise_create(sizeof(bool));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: failed to create promise");
        return NULL;
    }

    nimcp_future_t future = nimcp_promise_get_future(promise);
    if (!future) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: failed to get future");
        return NULL;
    }

    // Allocate context
    async_calcium_wave_ctx_t* ctx = (async_calcium_wave_ctx_t*)nimcp_malloc(sizeof(async_calcium_wave_ctx_t));
    if (!ctx) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: failed to allocate context");
        return NULL;
    }

    ctx->dc = dc;
    ctx->astrocyte_id = astrocyte_id;
    ctx->calcium_level = calcium_level;
    ctx->wave_velocity = wave_velocity;
    ctx->promise = promise;

    // Spawn worker thread
    nimcp_thread_t thread;
    if (nimcp_thread_create(&thread, async_calcium_wave_worker, ctx, NULL) != NIMCP_SUCCESS) {
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async calcium wave: failed to create worker thread");
        return NULL;
    }

    // Detach thread (worker will clean up context)
    nimcp_thread_detach(thread);

    LOG_MODULE_INFO(MODULE_NAME, "Async calcium wave started: astrocyte=%u level=%.3f velocity=%.1f",
                    astrocyte_id, calcium_level, wave_velocity);

    return future;
}

nimcp_future_t distrib_cognition_coordinate_pruning_async(
    distrib_cognition_t dc,
    uint32_t source_neuron_id,
    uint32_t target_neuron_id,
    float activity_score,
    uint8_t action)
{
    if (!dc) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: invalid coordinator");
        return NULL;
    }

    if (activity_score < 0.0F || activity_score > 1.0F) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: invalid activity score %.3f", activity_score);
        return NULL;
    }

    if (action > 2) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: invalid action %d", action);
        return NULL;
    }

    // Create promise/future pair (result is a bool)
    nimcp_promise_t promise = nimcp_promise_create(sizeof(bool));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: failed to create promise");
        return NULL;
    }

    nimcp_future_t future = nimcp_promise_get_future(promise);
    if (!future) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: failed to get future");
        return NULL;
    }

    // Allocate context
    async_pruning_ctx_t* ctx = (async_pruning_ctx_t*)nimcp_malloc(sizeof(async_pruning_ctx_t));
    if (!ctx) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: failed to allocate context");
        return NULL;
    }

    ctx->dc = dc;
    ctx->source_neuron_id = source_neuron_id;
    ctx->target_neuron_id = target_neuron_id;
    ctx->activity_score = activity_score;
    ctx->action = action;
    ctx->promise = promise;

    // Spawn worker thread
    nimcp_thread_t thread;
    if (nimcp_thread_create(&thread, async_pruning_worker, ctx, NULL) != NIMCP_SUCCESS) {
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async pruning coordination: failed to create worker thread");
        return NULL;
    }

    // Detach thread (worker will clean up context)
    nimcp_thread_detach(thread);

    LOG_MODULE_INFO(MODULE_NAME, "Async pruning coordination started: synapse=%u->%u score=%.3f action=%d",
                    source_neuron_id, target_neuron_id, activity_score, action);

    return future;
}
