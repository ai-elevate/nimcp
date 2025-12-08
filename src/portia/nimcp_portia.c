//=============================================================================
// nimcp_portia.c - Portia Spider Adaptive Intelligence Implementation
//=============================================================================
/**
 * @file nimcp_portia.c
 * @brief Portia system implementation
 *
 * WHAT: Dynamic resource optimization and platform adaptation
 * WHY:  Enable NIMCP to intelligently adapt to varying hardware constraints
 * HOW:  Monitor resources, adjust tier, handle degradation, coordinate via bio-async
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include "portia/nimcp_portia.h"
#include "portia/nimcp_portia_messages.h"
#include "async/nimcp_bio_router.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#define LOG_MODULE "portia"

//=============================================================================
// Subsystem Structures
//=============================================================================

/**
 * @brief Tier manager subsystem
 */
struct portia_tier_manager_t {
    platform_tier_t current_tier;
    platform_tier_t recommended_tier;
    portia_tier_config_t config;
    uint64_t last_switch_time_us;
    uint64_t tier_switch_count;
    nimcp_mutex_t lock;
};

/**
 * @brief Power monitor subsystem
 */
struct portia_power_monitor_t {
    portia_power_state_t current_state;
    portia_power_config_t config;
    float battery_level;
    bool is_on_ac;
    uint64_t last_poll_time_us;
    nimcp_mutex_t lock;
};

/**
 * @brief Resource tracker subsystem
 */
struct portia_resource_tracker_t {
    portia_resource_config_t config;
    system_resources_t current_resources;
    float cpu_usage;
    float memory_usage;
    float temperature_celsius;
    portia_thermal_state_t thermal_state;
    uint64_t last_sample_time_us;
    nimcp_mutex_t lock;
};

/**
 * @brief Degradation controller subsystem
 */
struct portia_degradation_controller_t {
    portia_degradation_level_t current_level;
    portia_degradation_config_t config;
    uint64_t degradation_count;
    uint64_t last_degradation_time_us;
    nimcp_mutex_t lock;
};

/**
 * @brief Accelerator detector subsystem
 */
struct portia_accelerator_detector_t {
    portia_accelerator_config_t config;
    uint32_t num_accelerators;
    portia_accelerator_type_t accelerator_types[8];
    bool detection_complete;
    nimcp_mutex_t lock;
};

/**
 * @brief Sensor fusion subsystem
 */
struct portia_sensor_fusion_t {
    float overall_health;
    float resource_pressure;
    float performance_score;
    float efficiency_score;
    uint64_t last_fusion_time_us;
    nimcp_mutex_t lock;
};

/**
 * @brief Planning engine subsystem
 */
struct portia_planning_engine_t {
    portia_workload_type_t current_workload;
    platform_tier_t planned_tier;
    bool planning_active;
    nimcp_mutex_t lock;
};

/**
 * @brief Target classifier subsystem
 */
struct portia_target_classifier_t {
    portia_workload_type_t classified_workload;
    float classification_confidence;
    uint32_t pattern_id;
    nimcp_mutex_t lock;
};

/**
 * @brief Main Portia context
 */
struct portia_context_t {
    portia_config_t config;
    bool initialized;

    /* Subsystems */
    portia_tier_manager_t* tier_manager;
    portia_power_monitor_t* power_monitor;
    portia_resource_tracker_t* resource_tracker;
    portia_degradation_controller_t* degradation_controller;
    portia_accelerator_detector_t* accelerator_detector;
    portia_sensor_fusion_t* sensor_fusion;
    portia_planning_engine_t* planning_engine;
    portia_target_classifier_t* target_classifier;

    /* Statistics */
    uint64_t update_count;
    float total_update_time_ms;
    uint64_t last_update_time_us;

    /* Thread safety */
    nimcp_mutex_t lock;
};

//=============================================================================
// Global Context
//=============================================================================

static portia_context_t* g_portia_ctx = NULL;

//=============================================================================
// Forward Declarations
//=============================================================================

/* TODO: Re-enable when bio-router integration is complete */
/* static nimcp_error_t portia_message_handler(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data); */

static nimcp_error_t portia_tier_manager_create(portia_tier_manager_t** out_mgr, const portia_tier_config_t* config);
static void portia_tier_manager_destroy(portia_tier_manager_t* mgr);
static nimcp_error_t portia_tier_manager_update(portia_tier_manager_t* mgr, const system_resources_t* resources);

static nimcp_error_t portia_power_monitor_create(portia_power_monitor_t** out_mon, const portia_power_config_t* config);
static void portia_power_monitor_destroy(portia_power_monitor_t* mon);
static nimcp_error_t portia_power_monitor_update(portia_power_monitor_t* mon);

static nimcp_error_t portia_resource_tracker_create(portia_resource_tracker_t** out_tracker, const portia_resource_config_t* config);
static void portia_resource_tracker_destroy(portia_resource_tracker_t* tracker);
static nimcp_error_t portia_resource_tracker_update(portia_resource_tracker_t* tracker);

static nimcp_error_t portia_degradation_controller_create(portia_degradation_controller_t** out_ctrl, const portia_degradation_config_t* config);
static void portia_degradation_controller_destroy(portia_degradation_controller_t* ctrl);
static nimcp_error_t portia_degradation_controller_update(portia_degradation_controller_t* ctrl, const portia_status_t* status);

static nimcp_error_t portia_accelerator_detector_create(portia_accelerator_detector_t** out_det, const portia_accelerator_config_t* config);
static void portia_accelerator_detector_destroy(portia_accelerator_detector_t* det);
static nimcp_error_t portia_accelerator_detector_scan(portia_accelerator_detector_t* det);

static nimcp_error_t portia_sensor_fusion_create(portia_sensor_fusion_t** out_fusion);
static void portia_sensor_fusion_destroy(portia_sensor_fusion_t* fusion);
static nimcp_error_t portia_sensor_fusion_update(portia_sensor_fusion_t* fusion, const portia_status_t* status);

static nimcp_error_t portia_planning_engine_create(portia_planning_engine_t** out_planner);
static void portia_planning_engine_destroy(portia_planning_engine_t* planner);

static nimcp_error_t portia_target_classifier_create(portia_target_classifier_t** out_classifier);
static void portia_target_classifier_destroy(portia_target_classifier_t* classifier);

//=============================================================================
// Default Configuration
//=============================================================================

portia_config_t portia_get_default_config(void) {
    portia_config_t config = {0};

    /* Tier configuration */
    config.tier_config.enable_auto_switching = true;
    config.tier_config.switch_hysteresis_ms = 5000;
    config.tier_config.upgrade_threshold = 0.7f;
    config.tier_config.downgrade_threshold = 0.3f;
    config.tier_config.lock_tier = false;

    /* Power configuration */
    config.power_config.enable_battery_awareness = true;
    config.power_config.poll_interval_ms = 10000;
    config.power_config.low_battery_threshold = 0.2f;
    config.power_config.critical_battery_threshold = 0.05f;
    config.power_config.enable_ac_detection = true;

    /* Resource configuration */
    config.resource_config.sample_interval_ms = 1000;
    config.resource_config.history_size = 60;
    config.resource_config.cpu_threshold = 0.8f;
    config.resource_config.memory_threshold = 0.85f;
    config.resource_config.thermal_threshold = 80.0f;

    /* Degradation configuration */
    config.degradation_config.enable_graceful_degradation = true;
    config.degradation_config.max_degradation = PORTIA_DEGRADATION_SEVERE;
    config.degradation_config.recovery_delay_ms = 30000;
    config.degradation_config.recovery_threshold = 0.5f;

    /* Accelerator configuration */
    config.accelerator_config.enable_gpu_detection = true;
    config.accelerator_config.enable_npu_detection = true;
    config.accelerator_config.enable_auto_offload = true;
    config.accelerator_config.detection_timeout_ms = 5000;

    /* General settings */
    config.enable_bio_async = true;
    config.update_interval_ms = 1000;
    config.enable_logging = true;
    config.enable_metrics = true;

    return config;
}

//=============================================================================
// Main API Implementation
//=============================================================================

nimcp_error_t portia_init(const portia_config_t* config) {
    /* Validate not already initialized */
    if (g_portia_ctx != NULL) {
        LOG_ERROR(LOG_MODULE, "Portia already initialized");
        return NIMCP_PORTIA_ERROR_ALREADY_INITIALIZED;
    }

    /* Use default config if none provided */
    portia_config_t cfg = config ? *config : portia_get_default_config();

    /* Security validation */
    if (config && !bbb_check_pointer(config, "portia_init")) {
        LOG_ERROR(LOG_MODULE, "Invalid config pointer");
        return NIMCP_ERROR_INVALID_ARG;
    }

    /* Allocate context */
    portia_context_t* ctx = nimcp_calloc(1, sizeof(portia_context_t));
    if (!ctx) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate Portia context");
        return NIMCP_ERROR_OUT_OF_MEMORY;
    }

    ctx->config = cfg;
    nimcp_mutex_init(&ctx->lock, NULL);

    LOG_INFO(LOG_MODULE, "Initializing Portia adaptive intelligence system");

    /* Initialize tier manager */
    nimcp_error_t err = portia_tier_manager_create(&ctx->tier_manager, &cfg.tier_config);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create tier manager: %d", err);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize power monitor */
    err = portia_power_monitor_create(&ctx->power_monitor, &cfg.power_config);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create power monitor: %d", err);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize resource tracker */
    err = portia_resource_tracker_create(&ctx->resource_tracker, &cfg.resource_config);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create resource tracker: %d", err);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize degradation controller */
    err = portia_degradation_controller_create(&ctx->degradation_controller, &cfg.degradation_config);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create degradation controller: %d", err);
        portia_resource_tracker_destroy(ctx->resource_tracker);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize accelerator detector */
    err = portia_accelerator_detector_create(&ctx->accelerator_detector, &cfg.accelerator_config);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create accelerator detector: %d", err);
        portia_degradation_controller_destroy(ctx->degradation_controller);
        portia_resource_tracker_destroy(ctx->resource_tracker);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize sensor fusion */
    err = portia_sensor_fusion_create(&ctx->sensor_fusion);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create sensor fusion: %d", err);
        portia_accelerator_detector_destroy(ctx->accelerator_detector);
        portia_degradation_controller_destroy(ctx->degradation_controller);
        portia_resource_tracker_destroy(ctx->resource_tracker);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize planning engine */
    err = portia_planning_engine_create(&ctx->planning_engine);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create planning engine: %d", err);
        portia_sensor_fusion_destroy(ctx->sensor_fusion);
        portia_accelerator_detector_destroy(ctx->accelerator_detector);
        portia_degradation_controller_destroy(ctx->degradation_controller);
        portia_resource_tracker_destroy(ctx->resource_tracker);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Initialize target classifier */
    err = portia_target_classifier_create(&ctx->target_classifier);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create target classifier: %d", err);
        portia_planning_engine_destroy(ctx->planning_engine);
        portia_sensor_fusion_destroy(ctx->sensor_fusion);
        portia_accelerator_detector_destroy(ctx->accelerator_detector);
        portia_degradation_controller_destroy(ctx->degradation_controller);
        portia_resource_tracker_destroy(ctx->resource_tracker);
        portia_power_monitor_destroy(ctx->power_monitor);
        portia_tier_manager_destroy(ctx->tier_manager);
        nimcp_free(ctx);
        return err;
    }

    /* Register with bio-router if enabled */
    if (cfg.enable_bio_async) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_UNKNOWN,  /* Will be assigned by router */
            .module_name = "portia",
            .inbox_capacity = 0,  /* Use default */
            .user_data = ctx
        };
        bio_module_context_t bio_ctx = bio_router_register_module(&bio_info);
        if (!bio_ctx) {
            LOG_WARN(LOG_MODULE, "Failed to register with bio-router (continuing anyway)");
        } else {
            LOG_INFO(LOG_MODULE, "Registered with bio-router");
            /* TODO: Store bio_ctx in portia_context_t for later use */
        }
    }

    /* Run initial accelerator detection */
    err = portia_accelerator_detector_scan(ctx->accelerator_detector);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Accelerator detection failed: %d", err);
    }

    ctx->initialized = true;
    g_portia_ctx = ctx;

    /* Security audit - TODO: Add BBB audit logging when available */
    /* bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "Portia system initialized successfully"); */

    LOG_INFO(LOG_MODULE, "Portia initialization complete");
    return NIMCP_SUCCESS;
}

void portia_destroy(void) {
    if (!g_portia_ctx) {
        return;
    }

    portia_context_t* ctx = g_portia_ctx;
    g_portia_ctx = NULL;

    LOG_INFO(LOG_MODULE, "Shutting down Portia system");

    /* Unregister from bio-router */
    /* TODO: Store and use bio_module_context_t */
    /* if (ctx->config.enable_bio_async && ctx->bio_ctx) {
        bio_router_unregister_module(ctx->bio_ctx);
    } */

    /* Destroy subsystems */
    portia_target_classifier_destroy(ctx->target_classifier);
    portia_planning_engine_destroy(ctx->planning_engine);
    portia_sensor_fusion_destroy(ctx->sensor_fusion);
    portia_accelerator_detector_destroy(ctx->accelerator_detector);
    portia_degradation_controller_destroy(ctx->degradation_controller);
    portia_resource_tracker_destroy(ctx->resource_tracker);
    portia_power_monitor_destroy(ctx->power_monitor);
    portia_tier_manager_destroy(ctx->tier_manager);

    nimcp_mutex_destroy(&ctx->lock);
    nimcp_free(ctx);

    /* bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "Portia system destroyed"); */
    LOG_INFO(LOG_MODULE, "Portia shutdown complete");
}

bool portia_is_initialized(void) {
    return g_portia_ctx != NULL && g_portia_ctx->initialized;
}

nimcp_error_t portia_update(void) {
    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    portia_context_t* ctx = g_portia_ctx;
    uint64_t start_time = nimcp_time_get_us();

    nimcp_mutex_lock(&ctx->lock);

    /* Update resource tracker */
    nimcp_error_t err = portia_resource_tracker_update(ctx->resource_tracker);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Resource tracker update failed: %d", err);
    }

    /* Update power monitor */
    err = portia_power_monitor_update(ctx->power_monitor);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Power monitor update failed: %d", err);
    }

    /* Update tier manager */
    err = portia_tier_manager_update(ctx->tier_manager, &ctx->resource_tracker->current_resources);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Tier manager update failed: %d", err);
    }

    /* Update sensor fusion */
    portia_status_t status;
    portia_get_status(&status);
    err = portia_sensor_fusion_update(ctx->sensor_fusion, &status);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Sensor fusion update failed: %d", err);
    }

    /* Update degradation controller */
    err = portia_degradation_controller_update(ctx->degradation_controller, &status);
    if (err != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Degradation controller update failed: %d", err);
    }

    /* Update statistics */
    ctx->update_count++;
    uint64_t end_time = nimcp_time_get_us();
    float update_time_ms = (end_time - start_time) / 1000.0f;
    ctx->total_update_time_ms += update_time_ms;
    ctx->last_update_time_us = end_time;

    nimcp_mutex_unlock(&ctx->lock);

    LOG_DEBUG(LOG_MODULE, "Portia update complete (%.3f ms)", update_time_ms);
    return NIMCP_SUCCESS;
}

nimcp_error_t portia_get_status(portia_status_t* status) {
    if (!bbb_check_pointer(status, "portia_get_status")) {
        return NIMCP_ERROR_INVALID_ARG;
    }

    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    portia_context_t* ctx = g_portia_ctx;

    nimcp_mutex_lock(&ctx->lock);

    memset(status, 0, sizeof(portia_status_t));

    /* Tier info */
    status->current_tier = ctx->tier_manager->current_tier;
    status->tier_switches = ctx->tier_manager->tier_switch_count;

    /* Power info */
    status->power_state = ctx->power_monitor->current_state;
    status->battery_level = ctx->power_monitor->battery_level;

    /* Resource info */
    status->cpu_usage = ctx->resource_tracker->cpu_usage;
    status->memory_usage = ctx->resource_tracker->memory_usage;
    status->temperature_celsius = ctx->resource_tracker->temperature_celsius;
    status->thermal_state = ctx->resource_tracker->thermal_state;

    /* Degradation info */
    status->degradation_level = ctx->degradation_controller->current_level;
    status->degradations = ctx->degradation_controller->degradation_count;

    /* Accelerator info */
    status->num_accelerators = ctx->accelerator_detector->num_accelerators;
    memcpy(status->accelerator_types, ctx->accelerator_detector->accelerator_types,
           sizeof(portia_accelerator_type_t) * 8);

    /* Workload info */
    status->current_workload = ctx->target_classifier->classified_workload;

    /* Statistics */
    status->updates = ctx->update_count;
    status->avg_update_time_ms = ctx->update_count > 0 ?
        ctx->total_update_time_ms / ctx->update_count : 0.0f;

    nimcp_mutex_unlock(&ctx->lock);

    return NIMCP_SUCCESS;
}

nimcp_error_t portia_set_tier(platform_tier_t tier) {
    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    if (tier >= PLATFORM_TIER_COUNT) {
        return NIMCP_ERROR_INVALID_ARG;
    }

    portia_context_t* ctx = g_portia_ctx;
    portia_tier_manager_t* mgr = ctx->tier_manager;

    nimcp_mutex_lock(&mgr->lock);

    if (mgr->config.lock_tier) {
        nimcp_mutex_unlock(&mgr->lock);
        LOG_WARN(LOG_MODULE, "Tier switching is locked");
        return NIMCP_PORTIA_ERROR_TIER_LOCKED;
    }

    platform_tier_t old_tier = mgr->current_tier;
    mgr->current_tier = tier;
    mgr->tier_switch_count++;
    mgr->last_switch_time_us = nimcp_time_get_us();

    nimcp_mutex_unlock(&mgr->lock);

    LOG_INFO(LOG_MODULE, "Tier changed: %s → %s",
             platform_tier_get_name(old_tier),
             platform_tier_get_name(tier));

    /* TODO: Broadcast tier change if bio-async enabled */
    /* Would need bio_module_context_t stored in ctx */

    /* bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "Tier changed to %s", platform_tier_get_name(tier)); */

    return NIMCP_SUCCESS;
}

platform_tier_t portia_get_current_tier(void) {
    if (!portia_is_initialized()) {
        return PLATFORM_TIER_MINIMAL;
    }

    return g_portia_ctx->tier_manager->current_tier;
}

nimcp_error_t portia_set_degradation_level(portia_degradation_level_t level) {
    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    portia_context_t* ctx = g_portia_ctx;
    portia_degradation_controller_t* ctrl = ctx->degradation_controller;

    nimcp_mutex_lock(&ctrl->lock);

    portia_degradation_level_t old_level = ctrl->current_level;
    ctrl->current_level = level;
    ctrl->degradation_count++;
    ctrl->last_degradation_time_us = nimcp_time_get_us();

    nimcp_mutex_unlock(&ctrl->lock);

    LOG_INFO(LOG_MODULE, "Degradation level changed: %s → %s",
             portia_degradation_level_name(old_level),
             portia_degradation_level_name(level));

    /* TODO: Broadcast degradation event if bio-async enabled */

    return NIMCP_SUCCESS;
}

uint32_t portia_recommend_neuron_count(void) {
    if (!portia_is_initialized()) {
        return 1000;
    }

    portia_context_t* ctx = g_portia_ctx;

    /* Get base recommendation from tier */
    platform_tier_t tier = ctx->tier_manager->current_tier;
    uint32_t base_count = platform_tier_recommend_neuron_count(tier,
        &ctx->resource_tracker->current_resources);

    /* Apply degradation multiplier */
    float degradation_multiplier = 1.0f;
    switch (ctx->degradation_controller->current_level) {
        case PORTIA_DEGRADATION_NONE:     degradation_multiplier = 1.0f; break;
        case PORTIA_DEGRADATION_MINOR:    degradation_multiplier = 0.8f; break;
        case PORTIA_DEGRADATION_MODERATE: degradation_multiplier = 0.5f; break;
        case PORTIA_DEGRADATION_SEVERE:   degradation_multiplier = 0.25f; break;
        case PORTIA_DEGRADATION_EMERGENCY: degradation_multiplier = 0.1f; break;
    }

    /* Apply power state multiplier */
    float power_multiplier = 1.0f;
    switch (ctx->power_monitor->current_state) {
        case PORTIA_POWER_AC:             power_multiplier = 1.0f; break;
        case PORTIA_POWER_BATTERY_FULL:   power_multiplier = 1.0f; break;
        case PORTIA_POWER_BATTERY_MID:    power_multiplier = 0.7f; break;
        case PORTIA_POWER_BATTERY_LOW:    power_multiplier = 0.4f; break;
        case PORTIA_POWER_BATTERY_CRITICAL: power_multiplier = 0.2f; break;
        case PORTIA_POWER_UNKNOWN:        power_multiplier = 0.8f; break;
    }

    uint32_t recommended = (uint32_t)(base_count * degradation_multiplier * power_multiplier);

    /* Ensure minimum */
    if (recommended < 100) {
        recommended = 100;
    }

    LOG_DEBUG(LOG_MODULE, "Recommended neuron count: %u (base=%u, degrade=%.2f, power=%.2f)",
              recommended, base_count, degradation_multiplier, power_multiplier);

    return recommended;
}

nimcp_error_t portia_set_auto_switching(bool enable) {
    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    portia_tier_manager_t* mgr = g_portia_ctx->tier_manager;

    nimcp_mutex_lock(&mgr->lock);
    mgr->config.enable_auto_switching = enable;
    nimcp_mutex_unlock(&mgr->lock);

    LOG_INFO(LOG_MODULE, "Automatic tier switching %s", enable ? "enabled" : "disabled");

    return NIMCP_SUCCESS;
}

nimcp_error_t portia_get_accelerators(
    portia_accelerator_type_t* out_accelerators,
    uint32_t max_accelerators,
    uint32_t* out_count)
{
    if (!bbb_check_pointer(out_accelerators, "portia_get_accelerators") ||
        !bbb_check_pointer(out_count, "portia_get_accelerators")) {
        return NIMCP_ERROR_INVALID_ARG;
    }

    if (!portia_is_initialized()) {
        return NIMCP_PORTIA_ERROR_NOT_INITIALIZED;
    }

    portia_accelerator_detector_t* det = g_portia_ctx->accelerator_detector;

    nimcp_mutex_lock(&det->lock);

    uint32_t count = det->num_accelerators < max_accelerators ?
                     det->num_accelerators : max_accelerators;

    memcpy(out_accelerators, det->accelerator_types,
           count * sizeof(portia_accelerator_type_t));
    *out_count = count;

    nimcp_mutex_unlock(&det->lock);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Bio-Async Message Handler
//=============================================================================

/* TODO: Implement message handler when bio-router integration is complete */
/*
static nimcp_error_t portia_message_handler(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)user_data;  // Suppress unused parameter warning

    if (!bbb_check_pointer(msg, "portia_message_handler")) {
        return NIMCP_ERROR_INVALID_ARG;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    LOG_DEBUG(LOG_MODULE, "Received message type 0x%04X (size=%zu)", header->type, msg_size);

    // Handle different message types
    // Note: Cast portia message types to bio_message_type_t
    if (header->type == (bio_message_type_t)BIO_MSG_TYPE_PORTIA_STATUS_QUERY) {
        // Query status
        portia_status_t status;
        nimcp_error_t err = portia_get_status(&status);
        if (err == NIMCP_SUCCESS && response_promise) {
            bio_msg_portia_status_response_t response = {0};
            portia_msg_init_header(&response.header,
                                   BIO_MSG_TYPE_PORTIA_STATUS_RESPONSE,
                                   BIO_MODULE_UNKNOWN,
                                   header->source_module,
                                   sizeof(response));
            response.status = status;
            nimcp_bio_promise_complete_sized(response_promise, &response, sizeof(response));
        }
        return err;
    } else if (header->type == (bio_message_type_t)BIO_MSG_TYPE_PORTIA_TIER_QUERY) {
        // Query tier
        platform_tier_t tier = portia_get_current_tier();
        if (response_promise) {
            nimcp_bio_promise_complete_sized(response_promise, &tier, sizeof(tier));
        }
        return NIMCP_SUCCESS;
    } else {
        LOG_DEBUG(LOG_MODULE, "Unhandled message type 0x%04X", header->type);
        return NIMCP_SUCCESS;
    }
}
*/

//=============================================================================
// Utility Functions
//=============================================================================

const char* portia_power_state_name(portia_power_state_t state) {
    switch (state) {
        case PORTIA_POWER_AC: return "AC";
        case PORTIA_POWER_BATTERY_FULL: return "BATTERY_FULL";
        case PORTIA_POWER_BATTERY_MID: return "BATTERY_MID";
        case PORTIA_POWER_BATTERY_LOW: return "BATTERY_LOW";
        case PORTIA_POWER_BATTERY_CRITICAL: return "BATTERY_CRITICAL";
        case PORTIA_POWER_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

const char* portia_thermal_state_name(portia_thermal_state_t state) {
    switch (state) {
        case PORTIA_THERMAL_NOMINAL: return "NOMINAL";
        case PORTIA_THERMAL_WARM: return "WARM";
        case PORTIA_THERMAL_HOT: return "HOT";
        case PORTIA_THERMAL_THROTTLED: return "THROTTLED";
        case PORTIA_THERMAL_CRITICAL: return "CRITICAL";
        default: return "INVALID";
    }
}

const char* portia_accelerator_type_name(portia_accelerator_type_t type) {
    switch (type) {
        case PORTIA_ACCEL_NONE: return "NONE";
        case PORTIA_ACCEL_GPU: return "GPU";
        case PORTIA_ACCEL_NPU: return "NPU";
        case PORTIA_ACCEL_TPU: return "TPU";
        case PORTIA_ACCEL_DSP: return "DSP";
        case PORTIA_ACCEL_FPGA: return "FPGA";
        case PORTIA_ACCEL_ASIC: return "ASIC";
        default: return "INVALID";
    }
}

const char* portia_workload_type_name(portia_workload_type_t type) {
    switch (type) {
        case PORTIA_WORKLOAD_TRAINING: return "TRAINING";
        case PORTIA_WORKLOAD_INFERENCE: return "INFERENCE";
        case PORTIA_WORKLOAD_MONITORING: return "MONITORING";
        case PORTIA_WORKLOAD_IDLE: return "IDLE";
        case PORTIA_WORKLOAD_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

const char* portia_degradation_level_name(portia_degradation_level_t level) {
    switch (level) {
        case PORTIA_DEGRADATION_NONE: return "NONE";
        case PORTIA_DEGRADATION_MINOR: return "MINOR";
        case PORTIA_DEGRADATION_MODERATE: return "MODERATE";
        case PORTIA_DEGRADATION_SEVERE: return "SEVERE";
        case PORTIA_DEGRADATION_EMERGENCY: return "EMERGENCY";
        default: return "INVALID";
    }
}

//=============================================================================
// Subsystem Implementations (Simplified for Core Infrastructure)
//=============================================================================

/* Tier Manager */
static nimcp_error_t portia_tier_manager_create(portia_tier_manager_t** out_mgr, const portia_tier_config_t* config) {
    portia_tier_manager_t* mgr = nimcp_calloc(1, sizeof(portia_tier_manager_t));
    if (!mgr) return NIMCP_ERROR_OUT_OF_MEMORY;

    mgr->config = *config;
    mgr->current_tier = platform_tier_detect();
    mgr->recommended_tier = mgr->current_tier;
    nimcp_mutex_init(&mgr->lock, NULL);

    *out_mgr = mgr;
    LOG_INFO(LOG_MODULE, "Tier manager created (initial tier: %s)", platform_tier_get_name(mgr->current_tier));
    return NIMCP_SUCCESS;
}

static void portia_tier_manager_destroy(portia_tier_manager_t* mgr) {
    if (!mgr) return;
    nimcp_mutex_destroy(&mgr->lock);
    nimcp_free(mgr);
}

static nimcp_error_t portia_tier_manager_update(portia_tier_manager_t* mgr, const system_resources_t* resources) {
    if (!mgr || !resources) return NIMCP_ERROR_INVALID_ARG;

    /* Simplified: just detect tier based on current resources */
    platform_tier_t detected = platform_tier_detect();
    mgr->recommended_tier = detected;

    return NIMCP_SUCCESS;
}

/* Power Monitor */
static nimcp_error_t portia_power_monitor_create(portia_power_monitor_t** out_mon, const portia_power_config_t* config) {
    portia_power_monitor_t* mon = nimcp_calloc(1, sizeof(portia_power_monitor_t));
    if (!mon) return NIMCP_ERROR_OUT_OF_MEMORY;

    mon->config = *config;
    mon->current_state = PORTIA_POWER_UNKNOWN;
    mon->battery_level = -1.0f;
    mon->is_on_ac = true;
    nimcp_mutex_init(&mon->lock, NULL);

    *out_mon = mon;
    LOG_INFO(LOG_MODULE, "Power monitor created");
    return NIMCP_SUCCESS;
}

static void portia_power_monitor_destroy(portia_power_monitor_t* mon) {
    if (!mon) return;
    nimcp_mutex_destroy(&mon->lock);
    nimcp_free(mon);
}

static nimcp_error_t portia_power_monitor_update(portia_power_monitor_t* mon) {
    if (!mon) return NIMCP_ERROR_INVALID_ARG;

    /* Simplified: assume AC power for now */
    nimcp_mutex_lock(&mon->lock);
    mon->current_state = PORTIA_POWER_AC;
    mon->is_on_ac = true;
    mon->battery_level = 1.0f;
    nimcp_mutex_unlock(&mon->lock);

    return NIMCP_SUCCESS;
}

/* Resource Tracker */
static nimcp_error_t portia_resource_tracker_create(portia_resource_tracker_t** out_tracker, const portia_resource_config_t* config) {
    portia_resource_tracker_t* tracker = nimcp_calloc(1, sizeof(portia_resource_tracker_t));
    if (!tracker) return NIMCP_ERROR_OUT_OF_MEMORY;

    tracker->config = *config;
    tracker->thermal_state = PORTIA_THERMAL_NOMINAL;
    nimcp_mutex_init(&tracker->lock, NULL);

    *out_tracker = tracker;
    LOG_INFO(LOG_MODULE, "Resource tracker created");
    return NIMCP_SUCCESS;
}

static void portia_resource_tracker_destroy(portia_resource_tracker_t* tracker) {
    if (!tracker) return;
    nimcp_mutex_destroy(&tracker->lock);
    nimcp_free(tracker);
}

static nimcp_error_t portia_resource_tracker_update(portia_resource_tracker_t* tracker) {
    if (!tracker) return NIMCP_ERROR_INVALID_ARG;

    nimcp_mutex_lock(&tracker->lock);

    /* Query system resources */
    system_resources_query(&tracker->current_resources);

    /* Simple CPU/memory usage estimation */
    tracker->cpu_usage = 0.5f; /* Placeholder */
    /* Calculate memory usage from available and total */
    uint64_t used_ram_mb = tracker->current_resources.total_ram_mb -
                          tracker->current_resources.available_ram_mb;
    tracker->memory_usage = (float)used_ram_mb /
                           (float)tracker->current_resources.total_ram_mb;
    tracker->temperature_celsius = 50.0f; /* Placeholder */
    tracker->thermal_state = PORTIA_THERMAL_NOMINAL;

    nimcp_mutex_unlock(&tracker->lock);

    return NIMCP_SUCCESS;
}

/* Degradation Controller */
static nimcp_error_t portia_degradation_controller_create(portia_degradation_controller_t** out_ctrl, const portia_degradation_config_t* config) {
    portia_degradation_controller_t* ctrl = nimcp_calloc(1, sizeof(portia_degradation_controller_t));
    if (!ctrl) return NIMCP_ERROR_OUT_OF_MEMORY;

    ctrl->config = *config;
    ctrl->current_level = PORTIA_DEGRADATION_NONE;
    nimcp_mutex_init(&ctrl->lock, NULL);

    *out_ctrl = ctrl;
    LOG_INFO(LOG_MODULE, "Degradation controller created");
    return NIMCP_SUCCESS;
}

static void portia_degradation_controller_destroy(portia_degradation_controller_t* ctrl) {
    if (!ctrl) return;
    nimcp_mutex_destroy(&ctrl->lock);
    nimcp_free(ctrl);
}

static nimcp_error_t portia_degradation_controller_update(portia_degradation_controller_t* ctrl, const portia_status_t* status) {
    if (!ctrl || !status) return NIMCP_ERROR_INVALID_ARG;

    /* Simplified: maintain current degradation level */
    return NIMCP_SUCCESS;
}

/* Accelerator Detector */
static nimcp_error_t portia_accelerator_detector_create(portia_accelerator_detector_t** out_det, const portia_accelerator_config_t* config) {
    portia_accelerator_detector_t* det = nimcp_calloc(1, sizeof(portia_accelerator_detector_t));
    if (!det) return NIMCP_ERROR_OUT_OF_MEMORY;

    det->config = *config;
    det->num_accelerators = 0;
    det->detection_complete = false;
    nimcp_mutex_init(&det->lock, NULL);

    *out_det = det;
    LOG_INFO(LOG_MODULE, "Accelerator detector created");
    return NIMCP_SUCCESS;
}

static void portia_accelerator_detector_destroy(portia_accelerator_detector_t* det) {
    if (!det) return;
    nimcp_mutex_destroy(&det->lock);
    nimcp_free(det);
}

static nimcp_error_t portia_accelerator_detector_scan(portia_accelerator_detector_t* det) {
    if (!det) return NIMCP_ERROR_INVALID_ARG;

    nimcp_mutex_lock(&det->lock);

    /* Simplified: no accelerators detected for now */
    det->num_accelerators = 0;
    det->detection_complete = true;

    nimcp_mutex_unlock(&det->lock);

    LOG_INFO(LOG_MODULE, "Accelerator scan complete (found %u accelerators)", det->num_accelerators);
    return NIMCP_SUCCESS;
}

/* Sensor Fusion */
static nimcp_error_t portia_sensor_fusion_create(portia_sensor_fusion_t** out_fusion) {
    portia_sensor_fusion_t* fusion = nimcp_calloc(1, sizeof(portia_sensor_fusion_t));
    if (!fusion) return NIMCP_ERROR_OUT_OF_MEMORY;

    fusion->overall_health = 1.0f;
    fusion->resource_pressure = 0.0f;
    fusion->performance_score = 1.0f;
    fusion->efficiency_score = 1.0f;
    nimcp_mutex_init(&fusion->lock, NULL);

    *out_fusion = fusion;
    LOG_INFO(LOG_MODULE, "Sensor fusion created");
    return NIMCP_SUCCESS;
}

static void portia_sensor_fusion_destroy(portia_sensor_fusion_t* fusion) {
    if (!fusion) return;
    nimcp_mutex_destroy(&fusion->lock);
    nimcp_free(fusion);
}

static nimcp_error_t portia_sensor_fusion_update(portia_sensor_fusion_t* fusion, const portia_status_t* status) {
    if (!fusion || !status) return NIMCP_ERROR_INVALID_ARG;

    nimcp_mutex_lock(&fusion->lock);

    /* Simplified fusion logic */
    fusion->resource_pressure = (status->cpu_usage + status->memory_usage) / 2.0f;
    fusion->overall_health = 1.0f - (fusion->resource_pressure * 0.5f);
    fusion->performance_score = 1.0f - (status->degradation_level * 0.2f);
    fusion->efficiency_score = fusion->overall_health * fusion->performance_score;

    nimcp_mutex_unlock(&fusion->lock);

    return NIMCP_SUCCESS;
}

/* Planning Engine */
static nimcp_error_t portia_planning_engine_create(portia_planning_engine_t** out_planner) {
    portia_planning_engine_t* planner = nimcp_calloc(1, sizeof(portia_planning_engine_t));
    if (!planner) return NIMCP_ERROR_OUT_OF_MEMORY;

    planner->current_workload = PORTIA_WORKLOAD_UNKNOWN;
    planner->planning_active = false;
    nimcp_mutex_init(&planner->lock, NULL);

    *out_planner = planner;
    LOG_INFO(LOG_MODULE, "Planning engine created");
    return NIMCP_SUCCESS;
}

static void portia_planning_engine_destroy(portia_planning_engine_t* planner) {
    if (!planner) return;
    nimcp_mutex_destroy(&planner->lock);
    nimcp_free(planner);
}

/* Target Classifier */
static nimcp_error_t portia_target_classifier_create(portia_target_classifier_t** out_classifier) {
    portia_target_classifier_t* classifier = nimcp_calloc(1, sizeof(portia_target_classifier_t));
    if (!classifier) return NIMCP_ERROR_OUT_OF_MEMORY;

    classifier->classified_workload = PORTIA_WORKLOAD_UNKNOWN;
    classifier->classification_confidence = 0.0f;
    nimcp_mutex_init(&classifier->lock, NULL);

    *out_classifier = classifier;
    LOG_INFO(LOG_MODULE, "Target classifier created");
    return NIMCP_SUCCESS;
}

static void portia_target_classifier_destroy(portia_target_classifier_t* classifier) {
    if (!classifier) return;
    nimcp_mutex_destroy(&classifier->lock);
    nimcp_free(classifier);
}
