//=============================================================================
// nimcp_portia_tier_switch.c - Dynamic Tier Switching Implementation
//=============================================================================

#include "portia/nimcp_portia_tier_switch.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>

#define LOG_MODULE "portia_tier_switch"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

//=============================================================================
// Constants
//=============================================================================

#define PORTIA_TIER_SWITCH_MAGIC 0x54495357  // 'TISW'
#define PORTIA_MODULE_NAME "portia_tier_switch"
#define PORTIA_MAX_CALLBACKS 8
#define PORTIA_MEMORY_SAFETY_MARGIN 0.15f  // Keep 15% RAM free
#define PORTIA_DEFAULT_EVALUATION_INTERVAL_MS 5000

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Callback registration entry
 */
typedef struct {
    tier_switch_callback_t callback;
    void* user_data;
    bool active;
} callback_entry_t;

/**
 * @brief Internal tier switch manager structure
 */
struct portia_tier_switch_struct {
    uint32_t magic;                     // Validation magic

    // Configuration
    tier_switch_config_t config;

    // State
    tier_switch_state_t state;

    // Thread synchronization
    nimcp_platform_mutex_t state_mutex;
    nimcp_thread_t monitor_thread;
    bool monitor_running;
    bool monitor_thread_created;

    // Callbacks
    callback_entry_t callbacks[PORTIA_MAX_CALLBACKS];
    nimcp_platform_mutex_t callback_mutex;
    bool callback_mutex_init;

    // Bio-async integration
    void* bio_ctx;
    bool bio_async_registered;
    uint32_t bio_module_id;

    // BBB integration
    bbb_system_t bbb_system;
    bool security_enabled;

    // Cached system resources
    system_resources_t last_resources;
    uint64_t last_resource_query_ms;

    // Statistics
    uint64_t creation_time_ms;
    uint64_t total_evaluation_time_ms;
    uint64_t total_transition_time_ms;
};

//=============================================================================
// Helper Functions - Time
//=============================================================================

static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

//=============================================================================
// Helper Functions - Validation
//=============================================================================

static bool validate_switcher(portia_tier_switch_t switcher) {
    if (!bbb_validate_pointer(NULL, switcher, sizeof(struct portia_tier_switch_struct), NULL)) {
        LOG_ERROR("[%s] Invalid switcher pointer", PORTIA_MODULE_NAME);
        return false;
    }

    if (switcher->magic != PORTIA_TIER_SWITCH_MAGIC) {
        LOG_ERROR("[%s] Invalid switcher magic: 0x%08X", PORTIA_MODULE_NAME, switcher->magic);
        bbb_audit_log(BBB_AUDIT_WARNING, LOG_MODULE, "invalid_magic", "invalid switcher magic");
        return false;
    }

    return true;
}

static bool validate_tier(platform_tier_t tier) {
    if (tier >= PLATFORM_TIER_COUNT) {
        LOG_ERROR("[%s] Invalid tier: %d", PORTIA_MODULE_NAME, tier);
        return false;
    }
    return true;
}

static bool validate_config(const tier_switch_config_t* config) {
    if (!config) {
        LOG_ERROR("[%s] Invalid config pointer", PORTIA_MODULE_NAME);
        return false;
    }

    // Validate threshold ranges
    if (config->memory_high_threshold < 0.0F || config->memory_high_threshold > 100.0F) {
        LOG_ERROR("[%s] Invalid memory_high_threshold: %.2f", PORTIA_MODULE_NAME, config->memory_high_threshold);
        return false;
    }

    if (config->memory_low_threshold < 0.0F || config->memory_low_threshold > 100.0F) {
        LOG_ERROR("[%s] Invalid memory_low_threshold: %.2f", PORTIA_MODULE_NAME, config->memory_low_threshold);
        return false;
    }

    if (config->memory_low_threshold >= config->memory_high_threshold) {
        LOG_ERROR("[%s] memory_low_threshold must be < memory_high_threshold", PORTIA_MODULE_NAME);
        return false;
    }

    LOG_DEBUG("[%s] Configuration validated successfully", PORTIA_MODULE_NAME);
    return true;
}

//=============================================================================
// Helper Functions - System Metrics
//=============================================================================

static bool query_system_metrics(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return false;
    }

    // Check cache freshness (cache for 1 second)
    uint64_t now = get_time_ms();
    if (now - switcher->last_resource_query_ms < 1000) {
        LOG_DEBUG("[%s] Using cached resource metrics", PORTIA_MODULE_NAME);
        return true;
    }

    // Query system resources
    system_resources_t resources;
    if (!system_resources_query(&resources)) {
        LOG_ERROR("[%s] Failed to query system resources", PORTIA_MODULE_NAME);
        return false;
    }

    // Update cached resources
    memcpy(&switcher->last_resources, &resources, sizeof(system_resources_t));
    switcher->last_resource_query_ms = now;

    // Calculate metrics
    nimcp_platform_mutex_lock(&switcher->state_mutex);

    if (resources.total_ram_mb > 0) {
        uint64_t used_ram = resources.total_ram_mb - resources.available_ram_mb;
        switcher->state.current_memory_usage_pct =
            (float)used_ram / (float)resources.total_ram_mb * 100.0F;
    }

    // Note: CPU temperature and battery require platform-specific APIs
    // For now, we'll use placeholder values (0.0 = not available)
    switcher->state.current_cpu_temp_c = 0.0F;  // TODO: Implement thermal monitoring
    switcher->state.current_battery_pct = 100.0F;  // TODO: Implement battery monitoring

    // CPU load (simple approximation based on thread count vs cores)
    if (resources.num_cpu_cores > 0) {
        // This is a placeholder - real implementation would query /proc/stat or similar
        switcher->state.current_cpu_load_pct = 0.0F;  // TODO: Implement CPU load monitoring
    }

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    LOG_DEBUG("[%s] System metrics: Memory=%.1f%%, Temp=%.1f°C, Battery=%.1f%%, Load=%.1f%%",
              PORTIA_MODULE_NAME,
              switcher->state.current_memory_usage_pct,
              switcher->state.current_cpu_temp_c,
              switcher->state.current_battery_pct,
              switcher->state.current_cpu_load_pct);

    return true;
}

//=============================================================================
// Helper Functions - Bio-Async Integration
//=============================================================================

static void broadcast_tier_switch_event(
    portia_tier_switch_t switcher,
    const tier_switch_event_t* event)
{
    if (!switcher->config.broadcast_events || !switcher->bio_async_registered) {
        return;
    }

    if (!switcher->config.bio_ctx) {
        LOG_WARN("[%s] Bio-async context not available for event broadcast", PORTIA_MODULE_NAME);
        return;
    }

    LOG_INFO("[%s] Broadcasting tier switch event: %s -> %s (trigger: %s)",
             PORTIA_MODULE_NAME,
             platform_tier_get_name(event->old_tier),
             platform_tier_get_name(event->new_tier),
             portia_tier_switch_trigger_name(event->trigger));

    // Create bio-async message (using system message type)
    // In a full implementation, this would use nimcp_bio_send_message() or similar
    // For now, we log the event
    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "tier_switch_event", "%s", event->reason);
}

//=============================================================================
// Helper Functions - Callbacks
//=============================================================================

static void invoke_callbacks(
    portia_tier_switch_t switcher,
    const tier_switch_event_t* event)
{
    if (!switcher->callback_mutex_init) {
        return;
    }

    nimcp_platform_mutex_lock(&switcher->callback_mutex);

    for (int i = 0; i < PORTIA_MAX_CALLBACKS; i++) {
        if (switcher->callbacks[i].active && switcher->callbacks[i].callback) {
            LOG_DEBUG("[%s] Invoking callback %d", PORTIA_MODULE_NAME, i);
            switcher->callbacks[i].callback(event, switcher->callbacks[i].user_data);
        }
    }

    nimcp_platform_mutex_unlock(&switcher->callback_mutex);
}

//=============================================================================
// Helper Functions - Tier Logic
//=============================================================================

static platform_tier_t calculate_downgrade_tier(
    portia_tier_switch_t switcher,
    tier_switch_trigger_t trigger)
{
    platform_tier_t current = switcher->state.current_tier;

    // Emergency conditions go to minimal
    if (trigger == TIER_SWITCH_TRIGGER_MEMORY_PRESSURE &&
        switcher->state.current_memory_usage_pct > switcher->config.memory_critical_threshold) {
        LOG_WARN("[%s] Critical memory pressure, downgrading to MINIMAL", PORTIA_MODULE_NAME);
        return PLATFORM_TIER_MINIMAL;
    }

    // Normal downgrade: go down one tier
    switch (current) {
        case PLATFORM_TIER_FULL:
            return PLATFORM_TIER_MEDIUM;
        case PLATFORM_TIER_MEDIUM:
            return PLATFORM_TIER_CONSTRAINED;
        case PLATFORM_TIER_CONSTRAINED:
            return PLATFORM_TIER_MINIMAL;
        case PLATFORM_TIER_MINIMAL:
            return PLATFORM_TIER_MINIMAL;  // Already at minimum
        default:
            return current;
    }
}

static platform_tier_t calculate_upgrade_tier(portia_tier_switch_t switcher) {
    platform_tier_t current = switcher->state.current_tier;

    // Upgrade one tier at a time
    switch (current) {
        case PLATFORM_TIER_MINIMAL:
            return PLATFORM_TIER_CONSTRAINED;
        case PLATFORM_TIER_CONSTRAINED:
            return PLATFORM_TIER_MEDIUM;
        case PLATFORM_TIER_MEDIUM:
            return PLATFORM_TIER_FULL;
        case PLATFORM_TIER_FULL:
            return PLATFORM_TIER_FULL;  // Already at maximum
        default:
            return current;
    }
}

//=============================================================================
// Monitoring Thread
//=============================================================================

static void* monitoring_thread_func(void* arg) {
    portia_tier_switch_t switcher = (portia_tier_switch_t)arg;

    LOG_INFO("[%s] Monitoring thread started", PORTIA_MODULE_NAME);

    while (switcher->monitor_running) {
        // Sleep for evaluation interval
        usleep(switcher->config.evaluation_interval_ms * 1000);

        if (!switcher->monitor_running) {
            break;
        }

        // Evaluate tier switch need
        platform_tier_t target_tier;
        tier_switch_trigger_t trigger;

        if (portia_tier_switch_evaluate(switcher, &target_tier, &trigger)) {
            LOG_INFO("[%s] Auto-switch recommended: %s -> %s (trigger: %s)",
                     PORTIA_MODULE_NAME,
                     platform_tier_get_name(switcher->state.current_tier),
                     platform_tier_get_name(target_tier),
                     portia_tier_switch_trigger_name(trigger));

            // Execute switch
            int result = portia_tier_switch_execute(switcher, target_tier, trigger);
            if (result != 0) {
                LOG_ERROR("[%s] Auto-switch failed: %d", PORTIA_MODULE_NAME, result);
            }
        }
    }

    LOG_INFO("[%s] Monitoring thread stopped", PORTIA_MODULE_NAME);
    return NULL;
}

//=============================================================================
// Public API - Lifecycle
//=============================================================================

tier_switch_config_t portia_tier_switch_default_config(void) {
    tier_switch_config_t config;
    memset(&config, 0, sizeof(tier_switch_config_t));

    // Memory thresholds
    config.memory_high_threshold = 85.0F;
    config.memory_low_threshold = 60.0F;
    config.memory_critical_threshold = 95.0F;

    // Thermal thresholds
    config.thermal_threshold_c = 80.0F;
    config.thermal_safe_c = 65.0F;

    // Battery thresholds
    config.battery_threshold_pct = 20.0F;
    config.battery_safe_pct = 50.0F;

    // Load thresholds
    config.load_threshold = 90.0F;
    config.load_safe = 50.0F;

    // Timing
    config.hysteresis_ms = 30000;  // 30 seconds
    config.evaluation_interval_ms = 5000;  // 5 seconds
    config.transition_timeout_ms = 60000;  // 60 seconds

    // Features
    config.auto_switch_enabled = true;
    config.allow_upgrade = true;
    config.allow_downgrade = true;
    config.broadcast_events = true;
    config.emergency_downgrade = true;

    // Module coordination
    config.module_shutdown_timeout_ms = 10000;  // 10 seconds
    config.wait_for_module_ack = true;

    // Bio-async
    config.bio_ctx = NULL;
    config.event_channel = BIO_CHANNEL_SEROTONIN;  // Slow, deliberative changes

    return config;
}

portia_tier_switch_t portia_tier_switch_init(const tier_switch_config_t* config) {
    LOG_INFO("[%s] Initializing tier switching system", PORTIA_MODULE_NAME);

    // Use default config if none provided
    tier_switch_config_t default_config = portia_tier_switch_default_config();
    const tier_switch_config_t* use_config = config ? config : &default_config;

    // Validate configuration
    if (!validate_config(use_config)) {
        LOG_ERROR("[%s] Configuration validation failed", PORTIA_MODULE_NAME);
        return NULL;
    }

    // Allocate switcher
    portia_tier_switch_t switcher = (portia_tier_switch_t)nimcp_calloc(
        1, sizeof(struct portia_tier_switch_struct));
    if (!switcher) {
        LOG_ERROR("[%s] Failed to allocate switcher structure", PORTIA_MODULE_NAME);
        return NULL;
    }

    // Initialize magic
    switcher->magic = PORTIA_TIER_SWITCH_MAGIC;

    // Copy configuration
    memcpy(&switcher->config, use_config, sizeof(tier_switch_config_t));

    // Initialize state
    switcher->state.current_tier = platform_tier_detect();
    switcher->state.target_tier = switcher->state.current_tier;
    switcher->state.previous_tier = switcher->state.current_tier;
    switcher->state.last_trigger = TIER_SWITCH_TRIGGER_INIT;
    switcher->state.last_switch_time_ms = get_time_ms();
    switcher->state.last_evaluation_ms = 0;
    switcher->state.switch_count = 0;
    switcher->state.upgrade_count = 0;
    switcher->state.downgrade_count = 0;
    switcher->state.failed_switch_count = 0;
    switcher->state.switch_in_progress = false;
    switcher->state.auto_switch_active = use_config->auto_switch_enabled;
    switcher->state.emergency_mode = false;

    // Create mutexes
    if (nimcp_platform_mutex_init(&switcher->state_mutex, false) != 0) {
        LOG_ERROR("[%s] Failed to create state mutex", PORTIA_MODULE_NAME);
        nimcp_free(switcher);
        return NULL;
    }

    if (nimcp_platform_mutex_init(&switcher->callback_mutex, false) != 0) {
        LOG_ERROR("[%s] Failed to create callback mutex", PORTIA_MODULE_NAME);
        nimcp_platform_mutex_destroy(&switcher->state_mutex);
        nimcp_free(switcher);
        return NULL;
    }
    switcher->callback_mutex_init = true;

    // Initialize callbacks
    for (int i = 0; i < PORTIA_MAX_CALLBACKS; i++) {
        switcher->callbacks[i].active = false;
    }

    // Initialize bio-async integration
    switcher->bio_ctx = use_config->bio_ctx;
    switcher->bio_async_registered = false;
    switcher->bio_module_id = BIO_MODULE_UNKNOWN;

    // Initialize BBB integration
    switcher->bbb_system = NULL;  // Will be set by security system if available
    switcher->security_enabled = false;

    // Initialize timestamps
    switcher->creation_time_ms = get_time_ms();
    switcher->last_resource_query_ms = 0;

    // Query initial system resources
    if (!query_system_metrics(switcher)) {
        LOG_WARN("[%s] Failed to query initial system metrics", PORTIA_MODULE_NAME);
    }

    // Start monitoring thread if auto-switch enabled
    if (use_config->auto_switch_enabled) {
        switcher->monitor_running = true;
        if (nimcp_thread_create(&switcher->monitor_thread, monitoring_thread_func, switcher, NULL) != 0) {
            LOG_ERROR("[%s] Failed to create monitoring thread", PORTIA_MODULE_NAME);
            nimcp_platform_mutex_destroy(&switcher->callback_mutex);
            nimcp_platform_mutex_destroy(&switcher->state_mutex);
            nimcp_free(switcher);
            return NULL;
        }
        switcher->monitor_thread_created = true;
        LOG_INFO("[%s] Monitoring thread started", PORTIA_MODULE_NAME);
    }

    LOG_INFO("[%s] Tier switching initialized: tier=%s, auto_switch=%s",
             PORTIA_MODULE_NAME,
             platform_tier_get_name(switcher->state.current_tier),
             use_config->auto_switch_enabled ? "enabled" : "disabled");

    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "tier_switch_init", "success");

    return switcher;
}

void portia_tier_switch_shutdown(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return;
    }

    LOG_INFO("[%s] Shutting down tier switching system", PORTIA_MODULE_NAME);

    // Stop monitoring thread
    if (switcher->monitor_thread_created) {
        switcher->monitor_running = false;
        nimcp_thread_join(switcher->monitor_thread, NULL);
        LOG_DEBUG("[%s] Monitoring thread stopped", PORTIA_MODULE_NAME);
    }

    // Destroy mutexes
    if (switcher->callback_mutex_init) {
        nimcp_platform_mutex_destroy(&switcher->callback_mutex);
    }

    nimcp_platform_mutex_destroy(&switcher->state_mutex);

    // Clear magic
    switcher->magic = 0;

    // Free memory
    nimcp_free(switcher);

    LOG_INFO("[%s] Shutdown complete", PORTIA_MODULE_NAME);
    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "tier_switch_shutdown", "success");
}

//=============================================================================
// Public API - Evaluation and Decision
//=============================================================================

bool portia_tier_switch_evaluate(
    portia_tier_switch_t switcher,
    platform_tier_t* target_tier,
    tier_switch_trigger_t* trigger)
{
    if (!validate_switcher(switcher)) {
        return false;
    }

    if (!bbb_validate_pointer(NULL, target_tier, sizeof(platform_tier_t), NULL) ||
        !bbb_validate_pointer(NULL, trigger, sizeof(tier_switch_trigger_t), NULL)) {
        LOG_ERROR("[%s] Invalid output parameters", PORTIA_MODULE_NAME);
        return false;
    }

    uint64_t eval_start = get_time_ms();

    // Query current system metrics
    if (!query_system_metrics(switcher)) {
        LOG_ERROR("[%s] Failed to query system metrics", PORTIA_MODULE_NAME);
        return false;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);

    // Check if switch is already in progress
    if (switcher->state.switch_in_progress) {
        LOG_DEBUG("[%s] Switch already in progress, skipping evaluation", PORTIA_MODULE_NAME);
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return false;
    }

    platform_tier_t current = switcher->state.current_tier;
    uint64_t now = get_time_ms();
    uint64_t time_since_last_switch = now - switcher->state.last_switch_time_ms;

    // Update evaluation timestamp
    switcher->state.last_evaluation_ms = now;

    // Default: stay at current tier
    *target_tier = current;
    *trigger = TIER_SWITCH_TRIGGER_INIT;
    bool should_switch = false;

    // Check downgrade conditions (higher priority)
    if (switcher->config.allow_downgrade) {
        // Memory pressure
        if (switcher->state.current_memory_usage_pct > switcher->config.memory_high_threshold) {
            *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_MEMORY_PRESSURE);
            *trigger = TIER_SWITCH_TRIGGER_MEMORY_PRESSURE;
            should_switch = true;
            LOG_INFO("[%s] Memory pressure detected: %.1f%% > %.1f%%",
                     PORTIA_MODULE_NAME,
                     switcher->state.current_memory_usage_pct,
                     switcher->config.memory_high_threshold);
        }
        // Thermal throttling
        else if (switcher->state.current_cpu_temp_c > 0.0F &&
                 switcher->state.current_cpu_temp_c > switcher->config.thermal_threshold_c) {
            *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_THERMAL_THROTTLE);
            *trigger = TIER_SWITCH_TRIGGER_THERMAL_THROTTLE;
            should_switch = true;
            LOG_INFO("[%s] Thermal throttling: %.1f°C > %.1f°C",
                     PORTIA_MODULE_NAME,
                     switcher->state.current_cpu_temp_c,
                     switcher->config.thermal_threshold_c);
        }
        // Battery low
        else if (switcher->state.current_battery_pct < 100.0F &&
                 switcher->state.current_battery_pct < switcher->config.battery_threshold_pct) {
            *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_BATTERY_LOW);
            *trigger = TIER_SWITCH_TRIGGER_BATTERY_LOW;
            should_switch = true;
            LOG_INFO("[%s] Low battery: %.1f%% < %.1f%%",
                     PORTIA_MODULE_NAME,
                     switcher->state.current_battery_pct,
                     switcher->config.battery_threshold_pct);
        }
        // Load spike
        else if (switcher->state.current_cpu_load_pct > 0.0F &&
                 switcher->state.current_cpu_load_pct > switcher->config.load_threshold) {
            *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_LOAD_SPIKE);
            *trigger = TIER_SWITCH_TRIGGER_LOAD_SPIKE;
            should_switch = true;
            LOG_INFO("[%s] CPU load spike: %.1f%% > %.1f%%",
                     PORTIA_MODULE_NAME,
                     switcher->state.current_cpu_load_pct,
                     switcher->config.load_threshold);
        }
    }

    // Check upgrade conditions (lower priority, only if no downgrade needed)
    if (!should_switch && switcher->config.allow_upgrade && current < PLATFORM_TIER_FULL) {
        bool memory_ok = switcher->state.current_memory_usage_pct < switcher->config.memory_low_threshold;
        bool thermal_ok = switcher->state.current_cpu_temp_c == 0.0F ||
                          switcher->state.current_cpu_temp_c < switcher->config.thermal_safe_c;
        bool battery_ok = switcher->state.current_battery_pct == 100.0F ||
                          switcher->state.current_battery_pct > switcher->config.battery_safe_pct;
        bool load_ok = switcher->state.current_cpu_load_pct == 0.0F ||
                       switcher->state.current_cpu_load_pct < switcher->config.load_safe;

        if (memory_ok && thermal_ok && battery_ok && load_ok) {
            *target_tier = calculate_upgrade_tier(switcher);
            *trigger = TIER_SWITCH_TRIGGER_RESOURCE_AVAILABLE;
            should_switch = true;
            LOG_INFO("[%s] Resources available for upgrade", PORTIA_MODULE_NAME);
        }
    }

    // Check hysteresis (except for emergency conditions)
    if (should_switch && *trigger != TIER_SWITCH_TRIGGER_MEMORY_PRESSURE) {
        if (time_since_last_switch < switcher->config.hysteresis_ms) {
            LOG_DEBUG("[%s] Hysteresis active: %lu ms < %u ms",
                      PORTIA_MODULE_NAME,
                      (unsigned long)time_since_last_switch,
                      switcher->config.hysteresis_ms);
            should_switch = false;
        }
    }

    // Emergency downgrade bypasses hysteresis
    if (switcher->config.emergency_downgrade &&
        switcher->state.current_memory_usage_pct > switcher->config.memory_critical_threshold) {
        *target_tier = PLATFORM_TIER_MINIMAL;
        *trigger = TIER_SWITCH_TRIGGER_MEMORY_PRESSURE;
        should_switch = true;
        LOG_WARN("[%s] EMERGENCY: Critical memory pressure %.1f%%",
                 PORTIA_MODULE_NAME,
                 switcher->state.current_memory_usage_pct);
    }

    // Update evaluation time statistics
    uint64_t eval_time = get_time_ms() - eval_start;
    switcher->total_evaluation_time_ms += eval_time;

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    if (should_switch) {
        LOG_INFO("[%s] Evaluation complete: %s -> %s (trigger: %s, eval_time: %lu ms)",
                 PORTIA_MODULE_NAME,
                 platform_tier_get_name(current),
                 platform_tier_get_name(*target_tier),
                 portia_tier_switch_trigger_name(*trigger),
                 (unsigned long)eval_time);
    } else {
        LOG_DEBUG("[%s] Evaluation complete: staying at %s (eval_time: %lu ms)",
                  PORTIA_MODULE_NAME,
                  platform_tier_get_name(current),
                  (unsigned long)eval_time);
    }

    return should_switch;
}

int portia_tier_switch_execute(
    portia_tier_switch_t switcher,
    platform_tier_t target_tier,
    tier_switch_trigger_t trigger)
{
    if (!validate_switcher(switcher)) {
        return -1;
    }

    if (!validate_tier(target_tier)) {
        return -1;
    }

    uint64_t transition_start = get_time_ms();

    nimcp_platform_mutex_lock(&switcher->state_mutex);

    platform_tier_t current_tier = switcher->state.current_tier;

    // Check if already at target tier
    if (current_tier == target_tier) {
        LOG_DEBUG("[%s] Already at target tier: %s",
                  PORTIA_MODULE_NAME,
                  platform_tier_get_name(target_tier));
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return 0;
    }

    // Check if switch already in progress
    if (switcher->state.switch_in_progress) {
        LOG_WARN("[%s] Switch already in progress", PORTIA_MODULE_NAME);
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return -1;
    }

    // Mark switch in progress
    switcher->state.switch_in_progress = true;
    switcher->state.target_tier = target_tier;
    switcher->state.last_trigger = trigger;

    bool is_upgrade = target_tier < current_tier;  // Lower tier number = higher capability
    bool is_emergency = (trigger == TIER_SWITCH_TRIGGER_MEMORY_PRESSURE &&
                         switcher->state.current_memory_usage_pct > switcher->config.memory_critical_threshold);

    switcher->state.emergency_mode = is_emergency;

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    LOG_INFO("[%s] Executing tier switch: %s -> %s (trigger: %s, %s%s)",
             PORTIA_MODULE_NAME,
             platform_tier_get_name(current_tier),
             platform_tier_get_name(target_tier),
             portia_tier_switch_trigger_name(trigger),
             is_upgrade ? "UPGRADE" : "DOWNGRADE",
             is_emergency ? ", EMERGENCY" : "");

    // Create event for callbacks
    tier_switch_event_t event;
    event.old_tier = current_tier;
    event.new_tier = target_tier;
    event.trigger = trigger;
    event.timestamp_ms = get_time_ms();
    event.was_emergency = is_emergency;
    event.transition_time_ms = 0;
    snprintf(event.reason, sizeof(event.reason),
             "Tier switch %s -> %s: %s",
             platform_tier_get_name(current_tier),
             platform_tier_get_name(target_tier),
             portia_tier_switch_trigger_name(trigger));

    // Broadcast pre-switch event
    broadcast_tier_switch_event(switcher, &event);

    // In a full implementation, this would:
    // 1. Send bio-async messages to cognitive modules
    // 2. Wait for module acknowledgments
    // 3. Suspend/shutdown modules not in target tier
    // 4. Apply new platform_tier_config_t
    // 5. Initialize/resume modules in target tier
    // For now, we simulate this with a delay

    if (!is_emergency && switcher->config.wait_for_module_ack) {
        LOG_DEBUG("[%s] Waiting for module acknowledgments...", PORTIA_MODULE_NAME);
        usleep(100000);  // 100ms simulation delay
    }

    // Apply new tier configuration
    platform_tier_config_t new_config = platform_tier_get_config(target_tier);
    LOG_INFO("[%s] Applied tier configuration: max_neurons=%u, cognitive_modules=0x%08X",
             PORTIA_MODULE_NAME,
             new_config.max_neurons,
             new_config.cognitive_modules_enabled);

    // Update state
    nimcp_platform_mutex_lock(&switcher->state_mutex);

    switcher->state.previous_tier = current_tier;
    switcher->state.current_tier = target_tier;
    switcher->state.target_tier = target_tier;
    switcher->state.last_switch_time_ms = get_time_ms();
    switcher->state.switch_count++;

    if (is_upgrade) {
        switcher->state.upgrade_count++;
    } else {
        switcher->state.downgrade_count++;
    }

    switcher->state.switch_in_progress = false;
    switcher->state.emergency_mode = false;

    uint64_t transition_time = get_time_ms() - transition_start;
    switcher->total_transition_time_ms += transition_time;

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    // Update event with transition time
    event.transition_time_ms = (uint32_t)transition_time;

    // Broadcast post-switch event
    broadcast_tier_switch_event(switcher, &event);

    // Invoke callbacks
    invoke_callbacks(switcher, &event);

    LOG_INFO("[%s] Tier switch complete: %s -> %s (transition_time: %lu ms)",
             PORTIA_MODULE_NAME,
             platform_tier_get_name(current_tier),
             platform_tier_get_name(target_tier),
             (unsigned long)transition_time);

    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "tier_switch_success", "%s", event.reason);

    return 0;
}

bool portia_tier_switch_can_upgrade(
    portia_tier_switch_t switcher,
    platform_tier_t target_tier)
{
    if (!validate_switcher(switcher) || !validate_tier(target_tier)) {
        return false;
    }

    // Query system metrics
    if (!query_system_metrics(switcher)) {
        LOG_ERROR("[%s] Failed to query system metrics", PORTIA_MODULE_NAME);
        return false;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);

    platform_tier_t current = switcher->state.current_tier;

    // Can't upgrade if already at target or higher
    if (current <= target_tier) {
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return false;
    }

    // Check hysteresis
    uint64_t now = get_time_ms();
    uint64_t time_since_last = now - switcher->state.last_switch_time_ms;
    if (time_since_last < switcher->config.hysteresis_ms) {
        LOG_DEBUG("[%s] Hysteresis active, cannot upgrade yet", PORTIA_MODULE_NAME);
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return false;
    }

    // Check resources
    bool memory_ok = switcher->state.current_memory_usage_pct < switcher->config.memory_low_threshold;
    bool thermal_ok = switcher->state.current_cpu_temp_c == 0.0F ||
                      switcher->state.current_cpu_temp_c < switcher->config.thermal_safe_c;
    bool battery_ok = switcher->state.current_battery_pct == 100.0F ||
                      switcher->state.current_battery_pct > switcher->config.battery_safe_pct;
    bool load_ok = switcher->state.current_cpu_load_pct == 0.0F ||
                   switcher->state.current_cpu_load_pct < switcher->config.load_safe;

    bool can_upgrade = memory_ok && thermal_ok && battery_ok && load_ok;

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    if (can_upgrade) {
        LOG_DEBUG("[%s] Upgrade to %s is safe",
                  PORTIA_MODULE_NAME,
                  platform_tier_get_name(target_tier));
    } else {
        LOG_DEBUG("[%s] Upgrade blocked: memory=%s, thermal=%s, battery=%s, load=%s",
                  PORTIA_MODULE_NAME,
                  memory_ok ? "ok" : "high",
                  thermal_ok ? "ok" : "hot",
                  battery_ok ? "ok" : "low",
                  load_ok ? "ok" : "high");
    }

    return can_upgrade;
}

bool portia_tier_switch_can_downgrade(
    portia_tier_switch_t switcher,
    platform_tier_t* target_tier,
    tier_switch_trigger_t* trigger)
{
    if (!validate_switcher(switcher)) {
        return false;
    }

    if (!bbb_validate_pointer(NULL, target_tier, sizeof(platform_tier_t), NULL) ||
        !bbb_validate_pointer(NULL, trigger, sizeof(tier_switch_trigger_t), NULL)) {
        LOG_ERROR("[%s] Invalid output parameters", PORTIA_MODULE_NAME);
        return false;
    }

    // Query system metrics
    if (!query_system_metrics(switcher)) {
        LOG_ERROR("[%s] Failed to query system metrics", PORTIA_MODULE_NAME);
        return false;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);

    platform_tier_t current = switcher->state.current_tier;
    *target_tier = current;
    *trigger = TIER_SWITCH_TRIGGER_INIT;
    bool should_downgrade = false;

    // Can't downgrade if already at minimum
    if (current == PLATFORM_TIER_MINIMAL) {
        nimcp_platform_mutex_unlock(&switcher->state_mutex);
        return false;
    }

    // Check downgrade conditions
    if (switcher->state.current_memory_usage_pct > switcher->config.memory_high_threshold) {
        *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_MEMORY_PRESSURE);
        *trigger = TIER_SWITCH_TRIGGER_MEMORY_PRESSURE;
        should_downgrade = true;
    }
    else if (switcher->state.current_cpu_temp_c > 0.0F &&
             switcher->state.current_cpu_temp_c > switcher->config.thermal_threshold_c) {
        *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_THERMAL_THROTTLE);
        *trigger = TIER_SWITCH_TRIGGER_THERMAL_THROTTLE;
        should_downgrade = true;
    }
    else if (switcher->state.current_battery_pct < 100.0F &&
             switcher->state.current_battery_pct < switcher->config.battery_threshold_pct) {
        *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_BATTERY_LOW);
        *trigger = TIER_SWITCH_TRIGGER_BATTERY_LOW;
        should_downgrade = true;
    }
    else if (switcher->state.current_cpu_load_pct > 0.0F &&
             switcher->state.current_cpu_load_pct > switcher->config.load_threshold) {
        *target_tier = calculate_downgrade_tier(switcher, TIER_SWITCH_TRIGGER_LOAD_SPIKE);
        *trigger = TIER_SWITCH_TRIGGER_LOAD_SPIKE;
        should_downgrade = true;
    }

    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    if (should_downgrade) {
        LOG_DEBUG("[%s] Downgrade needed: %s -> %s (trigger: %s)",
                  PORTIA_MODULE_NAME,
                  platform_tier_get_name(current),
                  platform_tier_get_name(*target_tier),
                  portia_tier_switch_trigger_name(*trigger));
    }

    return should_downgrade;
}

//=============================================================================
// Public API - State Query
//=============================================================================

int portia_tier_switch_get_state(
    portia_tier_switch_t switcher,
    tier_switch_state_t* state)
{
    if (!validate_switcher(switcher)) {
        return -1;
    }

    if (!bbb_validate_pointer(NULL, state, sizeof(tier_switch_state_t), NULL)) {
        LOG_ERROR("[%s] Invalid state pointer", PORTIA_MODULE_NAME);
        return -1;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    memcpy(state, &switcher->state, sizeof(tier_switch_state_t));
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    return 0;
}

platform_tier_t portia_tier_switch_get_current_tier(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return PLATFORM_TIER_MINIMAL;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    platform_tier_t tier = switcher->state.current_tier;
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    return tier;
}

bool portia_tier_switch_is_transitioning(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return false;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    bool transitioning = switcher->state.switch_in_progress;
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    return transitioning;
}

int portia_tier_switch_get_statistics(
    portia_tier_switch_t switcher,
    uint32_t* total_switches,
    uint32_t* upgrades,
    uint32_t* downgrades,
    uint32_t* failed)
{
    if (!validate_switcher(switcher)) {
        return -1;
    }

    if (!bbb_validate_pointer(NULL, total_switches, sizeof(uint32_t), NULL) ||
        !bbb_validate_pointer(NULL, upgrades, sizeof(uint32_t), NULL) ||
        !bbb_validate_pointer(NULL, downgrades, sizeof(uint32_t), NULL) ||
        !bbb_validate_pointer(NULL, failed, sizeof(uint32_t), NULL)) {
        LOG_ERROR("[%s] Invalid output parameters", PORTIA_MODULE_NAME);
        return -1;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    *total_switches = switcher->state.switch_count;
    *upgrades = switcher->state.upgrade_count;
    *downgrades = switcher->state.downgrade_count;
    *failed = switcher->state.failed_switch_count;
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    return 0;
}

//=============================================================================
// Public API - Configuration
//=============================================================================

void portia_tier_switch_set_auto_switch(
    portia_tier_switch_t switcher,
    bool enabled)
{
    if (!validate_switcher(switcher)) {
        return;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    bool was_enabled = switcher->state.auto_switch_active;
    switcher->state.auto_switch_active = enabled;
    switcher->config.auto_switch_enabled = enabled;
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    LOG_INFO("[%s] Auto-switch %s -> %s",
             PORTIA_MODULE_NAME,
             was_enabled ? "enabled" : "disabled",
             enabled ? "enabled" : "disabled");

    // Note: In full implementation, would start/stop monitoring thread here
}

int portia_tier_switch_update_config(
    portia_tier_switch_t switcher,
    const tier_switch_config_t* config)
{
    if (!validate_switcher(switcher) || !validate_config(config)) {
        return -1;
    }

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    memcpy(&switcher->config, config, sizeof(tier_switch_config_t));
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    LOG_INFO("[%s] Configuration updated", PORTIA_MODULE_NAME);
    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "config_update", "success");

    return 0;
}

void portia_tier_switch_set_callback(
    portia_tier_switch_t switcher,
    tier_switch_callback_t callback,
    void* user_data)
{
    if (!validate_switcher(switcher)) {
        return;
    }

    if (!callback) {
        LOG_ERROR("[%s] Invalid callback pointer", PORTIA_MODULE_NAME);
        return;
    }

    nimcp_platform_mutex_lock(&switcher->callback_mutex);

    // Find first available slot
    for (int i = 0; i < PORTIA_MAX_CALLBACKS; i++) {
        if (!switcher->callbacks[i].active) {
            switcher->callbacks[i].callback = callback;
            switcher->callbacks[i].user_data = user_data;
            switcher->callbacks[i].active = true;
            LOG_DEBUG("[%s] Callback registered in slot %d", PORTIA_MODULE_NAME, i);
            nimcp_platform_mutex_unlock(&switcher->callback_mutex);
            return;
        }
    }

    nimcp_platform_mutex_unlock(&switcher->callback_mutex);
    LOG_WARN("[%s] No callback slots available", PORTIA_MODULE_NAME);
}

//=============================================================================
// Public API - Manual Control
//=============================================================================

int portia_tier_switch_request(
    portia_tier_switch_t switcher,
    platform_tier_t target_tier)
{
    if (!validate_switcher(switcher) || !validate_tier(target_tier)) {
        return -1;
    }

    LOG_INFO("[%s] Manual tier switch requested: %s -> %s",
             PORTIA_MODULE_NAME,
             platform_tier_get_name(switcher->state.current_tier),
             platform_tier_get_name(target_tier));

    return portia_tier_switch_execute(switcher, target_tier, TIER_SWITCH_TRIGGER_USER_REQUEST);
}

int portia_tier_switch_emergency_downgrade(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return -1;
    }

    LOG_WARN("[%s] EMERGENCY DOWNGRADE requested", PORTIA_MODULE_NAME);

    nimcp_platform_mutex_lock(&switcher->state_mutex);
    switcher->state.emergency_mode = true;
    nimcp_platform_mutex_unlock(&switcher->state_mutex);

    int result = portia_tier_switch_execute(
        switcher,
        PLATFORM_TIER_MINIMAL,
        TIER_SWITCH_TRIGGER_MEMORY_PRESSURE);

    bbb_audit_log(BBB_AUDIT_INFO, LOG_MODULE, "emergency_downgrade",
                  "%s", result == 0 ? "success" : "failed");

    return result;
}

//=============================================================================
// Public API - Utility Functions
//=============================================================================

const char* portia_tier_switch_trigger_name(tier_switch_trigger_t trigger) {
    static const char* names[] = {
        "MEMORY_PRESSURE",
        "THERMAL_THROTTLE",
        "BATTERY_LOW",
        "LOAD_SPIKE",
        "USER_REQUEST",
        "PERFORMANCE_GOAL",
        "RESOURCE_AVAILABLE",
        "INIT"
    };

    if (trigger >= 0 && trigger < TIER_SWITCH_TRIGGER_COUNT) {
        return names[trigger];
    }

    return "UNKNOWN";
}

void portia_tier_switch_print_state(portia_tier_switch_t switcher) {
    if (!validate_switcher(switcher)) {
        return;
    }

    tier_switch_state_t state;
    if (portia_tier_switch_get_state(switcher, &state) != 0) {
        printf("Failed to get state\n");
        return;
    }

    printf("\n=== Portia Tier Switching State ===\n");
    printf("Current Tier:       %s\n", platform_tier_get_name(state.current_tier));
    printf("Target Tier:        %s\n", platform_tier_get_name(state.target_tier));
    printf("Previous Tier:      %s\n", platform_tier_get_name(state.previous_tier));
    printf("Last Trigger:       %s\n", portia_tier_switch_trigger_name(state.last_trigger));
    printf("Switch In Progress: %s\n", state.switch_in_progress ? "YES" : "NO");
    printf("Auto-Switch:        %s\n", state.auto_switch_active ? "ENABLED" : "DISABLED");
    printf("Emergency Mode:     %s\n", state.emergency_mode ? "YES" : "NO");
    printf("\n--- Statistics ---\n");
    printf("Total Switches:     %u\n", state.switch_count);
    printf("Upgrades:           %u\n", state.upgrade_count);
    printf("Downgrades:         %u\n", state.downgrade_count);
    printf("Failed Switches:    %u\n", state.failed_switch_count);
    printf("\n--- Current Metrics ---\n");
    printf("Memory Usage:       %.1f%%\n", state.current_memory_usage_pct);
    printf("CPU Temperature:    %.1f°C\n", state.current_cpu_temp_c);
    printf("Battery Level:      %.1f%%\n", state.current_battery_pct);
    printf("CPU Load:           %.1f%%\n", state.current_cpu_load_pct);
    printf("==================================\n\n");
}
