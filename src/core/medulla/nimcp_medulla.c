/**
 * @file nimcp_medulla.c
 * @brief Medulla Oblongata Main Orchestrator Implementation
 *
 * WHAT: Implementation of medulla orchestrator coordinating vital functions
 * WHY:  Provides centralized coordination of arousal, protection, circadian, and coupling
 * HOW:  Manages subsystems, integrates with external systems, coordinates state changes
 *
 * @author NIMCP Development Team
 * @date 2025-12-17
 * @version 1.0.0
 */

#include "core/medulla/nimcp_medulla.h"
#include "utils/fault_tolerance/nimcp_health_monitor.h"
#include "utils/fault_tolerance/nimcp_recovery.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include "async/nimcp_bio_router.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal medulla structure
 */
struct medulla_struct {
    // Configuration
    medulla_config_t config;

    // Subsystems (Phase 1.2-1.5 - placeholders for now)
    arousal_state_t arousal;
    protective_cutoff_t protection;
    brainstem_coupling_t coupling;
    circadian_rhythm_t circadian;

    // External system connections
    health_monitor_t health_monitor;
    nimcp_recovery_t recovery_system;
    sleep_system_t sleep_wake;
    neuromodulator_system_t neuromodulators;

    // State
    medulla_state_t state;
    arousal_level_t arousal_level;
    protection_level_t protection_level;
    circadian_phase_t circadian_phase;

    // Current values
    float current_arousal;          // 0-1
    float current_circadian_time;   // 0-24 hours

    // Statistics
    medulla_stats_t stats;

    // Bio-async
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    // Thread safety
    nimcp_platform_mutex_t* mutex;

    // Timing
    uint64_t last_update_us;
    uint64_t start_time_us;
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * WHAT: Get current timestamp in microseconds
 * WHY:  Timing for updates and statistics
 * HOW:  Use platform time API
 */
static uint64_t get_timestamp_us(void) {
    return nimcp_time_get_us();
}

/**
 * WHAT: Classify arousal level from continuous value
 * WHY:  Convert 0-1 arousal to categorical level
 * HOW:  Threshold-based categorization
 */
static arousal_level_t classify_arousal_level(float arousal) {
    if (arousal < 0.1f) return AROUSAL_LEVEL_COMA;
    if (arousal < 0.3f) return AROUSAL_LEVEL_DEEP_SLEEP;
    if (arousal < 0.4f) return AROUSAL_LEVEL_LIGHT_SLEEP;
    if (arousal < 0.5f) return AROUSAL_LEVEL_DROWSY;
    if (arousal < 0.7f) return AROUSAL_LEVEL_AWAKE;
    if (arousal < 0.9f) return AROUSAL_LEVEL_ALERT;
    return AROUSAL_LEVEL_HYPERAROUSAL;
}

/**
 * WHAT: Determine circadian phase from time
 * WHY:  Map circadian time to phase category
 * HOW:  Time-based categorization (24-hour cycle)
 */
static circadian_phase_t classify_circadian_phase(float hours) {
    // Normalize to 0-24 range
    while (hours >= 24.0f) hours -= 24.0f;
    while (hours < 0.0f) hours += 24.0f;

    if (hours >= 6.0f && hours < 9.0f) return CIRCADIAN_PHASE_EARLY_MORNING;
    if (hours >= 9.0f && hours < 12.0f) return CIRCADIAN_PHASE_MORNING;
    if (hours >= 12.0f && hours < 15.0f) return CIRCADIAN_PHASE_AFTERNOON;
    if (hours >= 15.0f && hours < 18.0f) return CIRCADIAN_PHASE_EVENING;
    if (hours >= 18.0f && hours < 21.0f) return CIRCADIAN_PHASE_LATE_EVENING;
    if (hours >= 21.0f && hours < 24.0f) return CIRCADIAN_PHASE_NIGHT;
    if (hours >= 0.0f && hours < 3.0f) return CIRCADIAN_PHASE_DEEP_NIGHT;
    return CIRCADIAN_PHASE_PRE_DAWN;
}

/**
 * WHAT: Get target arousal for circadian phase
 * WHY:  Different phases have different arousal targets
 * HOW:  Phase-dependent target values
 */
static float get_circadian_arousal_target(circadian_phase_t phase) {
    switch (phase) {
        case CIRCADIAN_PHASE_EARLY_MORNING: return 0.4f;  // Rising
        case CIRCADIAN_PHASE_MORNING: return 0.7f;        // Peak
        case CIRCADIAN_PHASE_AFTERNOON: return 0.6f;      // Dip
        case CIRCADIAN_PHASE_EVENING: return 0.7f;        // Second peak
        case CIRCADIAN_PHASE_LATE_EVENING: return 0.5f;   // Declining
        case CIRCADIAN_PHASE_NIGHT: return 0.3f;          // Sleep prep
        case CIRCADIAN_PHASE_DEEP_NIGHT: return 0.2f;     // Minimum
        case CIRCADIAN_PHASE_PRE_DAWN: return 0.3f;       // Pre-wake
        default: return 0.5f;
    }
}

/**
 * WHAT: Determine protection level from health score
 * WHY:  Map health status to protection response
 * HOW:  Threshold-based escalation
 */
static protection_level_t determine_protection_level(float health_score) {
    if (health_score < 10.0f) return PROTECTION_LEVEL_SHUTDOWN;
    if (health_score < 30.0f) return PROTECTION_LEVEL_CRITICAL;
    if (health_score < 50.0f) return PROTECTION_LEVEL_DEFENSIVE;
    if (health_score < 70.0f) return PROTECTION_LEVEL_GUARDED;
    if (health_score < 90.0f) return PROTECTION_LEVEL_CAUTIOUS;
    return PROTECTION_LEVEL_NORMAL;
}

/**
 * WHAT: Modulate arousal based on protection level
 * WHY:  Reduce arousal during protective states
 * HOW:  Apply protection-based suppression factor
 */
static float apply_protection_modulation(float arousal, protection_level_t protection) {
    float factor = 1.0f;
    switch (protection) {
        case PROTECTION_LEVEL_CAUTIOUS: factor = 0.9f; break;
        case PROTECTION_LEVEL_GUARDED: factor = 0.7f; break;
        case PROTECTION_LEVEL_DEFENSIVE: factor = 0.5f; break;
        case PROTECTION_LEVEL_CRITICAL: factor = 0.3f; break;
        case PROTECTION_LEVEL_SHUTDOWN: factor = 0.1f; break;
        default: factor = 1.0f; break;
    }
    return arousal * factor;
}

/**
 * WHAT: Update circadian rhythm
 * WHY:  Progress circadian time
 * HOW:  Increment time, wrap at period
 */
static void update_circadian(medulla_t medulla, float dt) {
    if (!medulla || !medulla->config.circadian.enable_synchronization) {
        return;
    }

    // Progress time (hours)
    float dt_hours = dt / 3600.0f;
    medulla->current_circadian_time += dt_hours;

    // Wrap at period
    float period = medulla->config.circadian.period_hours;
    while (medulla->current_circadian_time >= period) {
        medulla->current_circadian_time -= period;
        medulla->stats.circadian_cycles++;
    }

    // Update phase
    medulla->circadian_phase = classify_circadian_phase(medulla->current_circadian_time);
    medulla->stats.circadian_time_hours = medulla->current_circadian_time;
}

/**
 * WHAT: Update arousal state
 * WHY:  Manage consciousness level
 * HOW:  Apply circadian target, protection modulation, decay
 */
static void update_arousal(medulla_t medulla, float dt) {
    if (!medulla) return;

    // Get circadian target
    float target = get_circadian_arousal_target(medulla->circadian_phase);

    // Move towards target
    float rate = medulla->config.arousal.arousal_decay_rate;
    float delta = target - medulla->current_arousal;
    medulla->current_arousal += delta * rate * dt;

    // Apply protection modulation
    medulla->current_arousal = apply_protection_modulation(
        medulla->current_arousal,
        medulla->protection_level
    );

    // Clamp
    if (medulla->current_arousal < medulla->config.arousal.min_arousal) {
        medulla->current_arousal = medulla->config.arousal.min_arousal;
    }
    if (medulla->current_arousal > medulla->config.arousal.max_arousal) {
        medulla->current_arousal = medulla->config.arousal.max_arousal;
    }

    // Update classification
    medulla->arousal_level = classify_arousal_level(medulla->current_arousal);

    // Update stats
    medulla->stats.arousal_updates++;
    medulla->stats.avg_arousal = (medulla->stats.avg_arousal * (medulla->stats.arousal_updates - 1) +
                                   medulla->current_arousal) / medulla->stats.arousal_updates;
}

/**
 * WHAT: Update protection level
 * WHY:  Respond to health status
 * HOW:  Query health monitor, adjust protection
 */
static void update_protection(medulla_t medulla) {
    if (!medulla || !medulla->health_monitor) {
        return;
    }

    // Get health score
    float health_score = health_monitor_get_score(medulla->health_monitor);

    // Determine new protection level
    protection_level_t new_level = determine_protection_level(health_score);

    // Track state changes
    if (new_level != medulla->protection_level) {
        if (new_level > PROTECTION_LEVEL_NORMAL) {
            medulla->stats.protection_activations++;
        }
        medulla->protection_level = new_level;
    }

    // Emergency shutdown if critical
    if (new_level == PROTECTION_LEVEL_SHUTDOWN &&
        medulla->config.protection.enable_auto_shutdown) {
        medulla_emergency_shutdown(medulla, "Health score critical");
    }
}

/**
 * WHAT: Synchronize with sleep-wake system
 * WHY:  Bidirectional arousal-sleep state coupling
 * HOW:  Query sleep state, adjust arousal accordingly
 */
static void synchronize_sleep_wake(medulla_t medulla) {
    if (!medulla || !medulla->sleep_wake) {
        return;
    }

    sleep_state_t sleep_state = sleep_get_current_state(medulla->sleep_wake);

    // Map sleep state to arousal
    float target_arousal = 0.5f;
    switch (sleep_state) {
        case SLEEP_STATE_AWAKE:
            target_arousal = 0.7f;
            break;
        case SLEEP_STATE_DROWSY:
            target_arousal = 0.4f;
            break;
        case SLEEP_STATE_LIGHT_NREM:
            target_arousal = 0.3f;
            break;
        case SLEEP_STATE_DEEP_NREM:
            target_arousal = 0.2f;
            break;
        case SLEEP_STATE_REM:
            target_arousal = 0.5f;  // REM has higher arousal
            break;
    }

    // Blend with current arousal
    medulla->current_arousal = medulla->current_arousal * 0.7f + target_arousal * 0.3f;
    medulla->stats.sleep_transitions++;
}

/**
 * WHAT: Apply neuromodulator adjustments
 * WHY:  Neuromodulators influence arousal and circadian
 * HOW:  Query neuromodulator levels, apply modulation
 */
static void apply_neuromodulator_modulation(medulla_t medulla) {
    if (!medulla || !medulla->neuromodulators) {
        return;
    }

    // Get norepinephrine (arousal)
    float norepinephrine = neuromodulator_get_level(
        medulla->neuromodulators,
        NEUROMOD_NOREPINEPHRINE
    );

    // Get acetylcholine (alertness)
    float acetylcholine = neuromodulator_get_level(
        medulla->neuromodulators,
        NEUROMOD_ACETYLCHOLINE
    );

    // Modulate arousal
    float boost = (norepinephrine + acetylcholine) * 0.1f;
    medulla->current_arousal += boost;

    // Clamp
    if (medulla->current_arousal > medulla->config.arousal.max_arousal) {
        medulla->current_arousal = medulla->config.arousal.max_arousal;
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

medulla_config_t medulla_default_config(void) {
    medulla_config_t config;
    memset(&config, 0, sizeof(config));

    // Arousal defaults
    config.arousal.baseline_arousal = 0.5f;
    config.arousal.arousal_decay_rate = 0.1f;
    config.arousal.min_arousal = 0.0f;
    config.arousal.max_arousal = 1.0f;
    config.arousal.enable_auto_regulation = true;

    // Protection defaults
    config.protection.health_threshold_critical = 30.0f;
    config.protection.health_threshold_defensive = 50.0f;
    config.protection.recovery_time_ms = 5000;
    config.protection.enable_auto_shutdown = true;

    // Coupling defaults
    config.coupling.coupling_strength = 0.8f;
    config.coupling.latency_ms = 10.0f;
    config.coupling.enable_bidirectional = true;

    // Circadian defaults
    config.circadian.period_hours = 24.0f;
    config.circadian.phase_offset_hours = 0.0f;
    config.circadian.amplitude = 0.5f;
    config.circadian.enable_synchronization = true;

    // Integration defaults
    config.enable_health_integration = true;
    config.enable_recovery_integration = true;
    config.enable_sleep_integration = true;
    config.enable_neuromod_integration = true;

    // Update timing
    config.update_interval_ms = 100;

    // Bio-async
    config.enable_bio_async = true;
    config.inbox_capacity = 64;

    return config;
}

medulla_t medulla_create(const medulla_config_t* config) {
    // Allocate structure
    medulla_t medulla = nimcp_calloc(1, sizeof(struct medulla_struct));
    if (!medulla) {
        NIMCP_LOGGING_ERROR("Failed to allocate medulla structure");
        return NULL;
    }

    // Copy config
    if (config) {
        medulla->config = *config;
    } else {
        medulla->config = medulla_default_config();
    }

    // Initialize state
    medulla->state = MEDULLA_STATE_STOPPED;
    medulla->arousal_level = AROUSAL_LEVEL_AWAKE;
    medulla->protection_level = PROTECTION_LEVEL_NORMAL;
    medulla->circadian_phase = CIRCADIAN_PHASE_MORNING;

    // Initialize values
    medulla->current_arousal = medulla->config.arousal.baseline_arousal;
    medulla->current_circadian_time = medulla->config.circadian.phase_offset_hours;

    // Initialize stats
    memset(&medulla->stats, 0, sizeof(medulla_stats_t));
    medulla->stats.state = medulla->state;
    medulla->stats.arousal_level = medulla->arousal_level;
    medulla->stats.protection_level = medulla->protection_level;
    medulla->stats.circadian_phase = medulla->circadian_phase;
    medulla->stats.current_arousal = medulla->current_arousal;

    // Create mutex
    medulla->mutex = nimcp_platform_mutex_create();
    if (!medulla->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create medulla mutex");
        nimcp_free(medulla);
        return NULL;
    }

    // Subsystems will be created in Phase 1.2-1.5
    medulla->arousal = NULL;
    medulla->protection = NULL;
    medulla->coupling = NULL;
    medulla->circadian = NULL;

    // External connections initialized to NULL
    medulla->health_monitor = NULL;
    medulla->recovery_system = NULL;
    medulla->sleep_wake = NULL;
    medulla->neuromodulators = NULL;

    // Bio-async
    medulla->bio_async_enabled = false;
    medulla->bio_ctx = NULL;

    // Timing
    medulla->last_update_us = get_timestamp_us();
    medulla->start_time_us = medulla->last_update_us;

    NIMCP_LOGGING_INFO("Medulla orchestrator created");
    return medulla;
}

void medulla_destroy(medulla_t medulla) {
    if (!medulla) {
        return;
    }

    // Stop if running
    if (medulla->state == MEDULLA_STATE_RUNNING) {
        medulla_stop(medulla);
    }

    // Disconnect bio-async
    if (medulla->bio_async_enabled) {
        medulla_disconnect_bio_async(medulla);
    }

    // Destroy subsystems (Phase 1.2-1.5 will implement)
    // arousal_state_destroy(medulla->arousal);
    // protective_cutoff_destroy(medulla->protection);
    // brainstem_coupling_destroy(medulla->coupling);
    // circadian_rhythm_destroy(medulla->circadian);

    // Destroy mutex
    if (medulla->mutex) {
        nimcp_platform_mutex_destroy(medulla->mutex);
    }

    // Free structure
    nimcp_free(medulla);

    NIMCP_LOGGING_INFO("Medulla orchestrator destroyed");
}

//=============================================================================
// Control Functions
//=============================================================================

int medulla_start(medulla_t medulla) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);

    if (medulla->state == MEDULLA_STATE_RUNNING) {
        nimcp_platform_mutex_unlock(medulla->mutex);
        return 0;  // Already running
    }

    medulla->state = MEDULLA_STATE_STARTING;

    // Start subsystems (Phase 1.2-1.5 will implement)
    // arousal_state_start(medulla->arousal);
    // protective_cutoff_start(medulla->protection);
    // brainstem_coupling_start(medulla->coupling);
    // circadian_rhythm_start(medulla->circadian);

    // Reset timing
    medulla->last_update_us = get_timestamp_us();
    medulla->start_time_us = medulla->last_update_us;

    medulla->state = MEDULLA_STATE_RUNNING;
    medulla->stats.state = medulla->state;

    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Medulla orchestrator started");
    return 0;
}

int medulla_stop(medulla_t medulla) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);

    if (medulla->state == MEDULLA_STATE_STOPPED) {
        nimcp_platform_mutex_unlock(medulla->mutex);
        return 0;  // Already stopped
    }

    medulla->state = MEDULLA_STATE_STOPPING;

    // Stop subsystems (Phase 1.2-1.5 will implement)
    // arousal_state_stop(medulla->arousal);
    // protective_cutoff_stop(medulla->protection);
    // brainstem_coupling_stop(medulla->coupling);
    // circadian_rhythm_stop(medulla->circadian);

    medulla->state = MEDULLA_STATE_STOPPED;
    medulla->stats.state = medulla->state;

    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Medulla orchestrator stopped");
    return 0;
}

int medulla_update(medulla_t medulla, float dt) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);

    if (medulla->state != MEDULLA_STATE_RUNNING) {
        nimcp_platform_mutex_unlock(medulla->mutex);
        return -1;
    }

    uint64_t start_us = get_timestamp_us();

    // Coordination pipeline
    // 1. Update circadian rhythm
    update_circadian(medulla, dt);

    // 2. Check health monitor and update protection
    if (medulla->config.enable_health_integration) {
        update_protection(medulla);
    }

    // 3. Update arousal state (modulated by protection/circadian)
    update_arousal(medulla, dt);

    // 4. Apply neuromodulator adjustments
    if (medulla->config.enable_neuromod_integration) {
        apply_neuromodulator_modulation(medulla);
    }

    // 5. Synchronize with sleep-wake
    if (medulla->config.enable_sleep_integration) {
        synchronize_sleep_wake(medulla);
    }

    // 6. Update brainstem coupling (Phase 1.4 will implement)
    // brainstem_coupling_update(medulla->coupling, dt);

    // Update stats
    medulla->stats.total_updates++;
    medulla->stats.current_arousal = medulla->current_arousal;
    medulla->stats.arousal_level = medulla->arousal_level;
    medulla->stats.protection_level = medulla->protection_level;
    medulla->stats.circadian_phase = medulla->circadian_phase;

    uint64_t end_us = get_timestamp_us();
    float update_time_us = (float)(end_us - start_us);
    medulla->stats.avg_update_time_us =
        (medulla->stats.avg_update_time_us * (medulla->stats.total_updates - 1) +
         update_time_us) / medulla->stats.total_updates;

    medulla->stats.uptime_ms = (end_us - medulla->start_time_us) / 1000;
    medulla->last_update_us = end_us;

    nimcp_platform_mutex_unlock(medulla->mutex);
    return 0;
}

int medulla_emergency_shutdown(medulla_t medulla, const char* reason) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);

    NIMCP_LOGGING_ERROR("Medulla emergency shutdown: %s", reason ? reason : "unknown");

    // Set emergency state
    medulla->state = MEDULLA_STATE_EMERGENCY;
    medulla->protection_level = PROTECTION_LEVEL_SHUTDOWN;
    medulla->current_arousal = medulla->config.arousal.min_arousal;

    // Update stats
    medulla->stats.emergency_shutdowns++;
    medulla->stats.state = medulla->state;
    medulla->stats.protection_level = medulla->protection_level;

    // Trigger recovery if connected
    if (medulla->recovery_system && medulla->config.enable_recovery_integration) {
        medulla->stats.recovery_triggers++;
        // nimcp_recovery_trigger(medulla->recovery_system, reason);
    }

    nimcp_platform_mutex_unlock(medulla->mutex);
    return 0;
}

int medulla_request_state_change(medulla_t medulla, medulla_state_t new_state) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);

    // Validate transition
    if (medulla->state == MEDULLA_STATE_EMERGENCY &&
        new_state != MEDULLA_STATE_STOPPED) {
        NIMCP_LOGGING_WARN("Cannot transition from EMERGENCY except to STOPPED");
        nimcp_platform_mutex_unlock(medulla->mutex);
        return -1;
    }

    medulla->state = new_state;
    medulla->stats.state = new_state;

    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Medulla state changed to %s",
                       medulla_state_to_string(new_state));
    return 0;
}

//=============================================================================
// Integration Functions
//=============================================================================

int medulla_connect_health_monitor(medulla_t medulla, health_monitor_t health_monitor) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->health_monitor = health_monitor;
    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Health monitor connected to medulla");
    return 0;
}

int medulla_connect_recovery_system(medulla_t medulla, nimcp_recovery_t recovery) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->recovery_system = recovery;
    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Recovery system connected to medulla");
    return 0;
}

int medulla_connect_sleep_wake(medulla_t medulla, sleep_system_t sleep_wake) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->sleep_wake = sleep_wake;
    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Sleep-wake system connected to medulla");
    return 0;
}

int medulla_connect_neuromodulators(medulla_t medulla, neuromodulator_system_t neuromodulators) {
    if (!medulla) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->neuromodulators = neuromodulators;
    nimcp_platform_mutex_unlock(medulla->mutex);

    NIMCP_LOGGING_INFO("Neuromodulator system connected to medulla");
    return 0;
}

//=============================================================================
// Query Functions
//=============================================================================

float medulla_get_arousal_level(const medulla_t medulla) {
    if (!medulla) {
        return -1.0f;
    }

    return medulla->current_arousal;
}

int medulla_boost_arousal(medulla_t medulla, float delta) {
    if (!medulla || delta < 0.0f) {
        return -1;
    }

    float new_arousal = medulla->current_arousal + delta;

    /* Clamp to configured range */
    if (new_arousal > medulla->config.arousal.max_arousal) {
        new_arousal = medulla->config.arousal.max_arousal;
    }

    medulla->current_arousal = new_arousal;
    return 0;
}

int medulla_reduce_arousal(medulla_t medulla, float delta) {
    if (!medulla || delta < 0.0f) {
        return -1;
    }

    float new_arousal = medulla->current_arousal - delta;

    /* Clamp to configured range */
    if (new_arousal < medulla->config.arousal.min_arousal) {
        new_arousal = medulla->config.arousal.min_arousal;
    }

    medulla->current_arousal = new_arousal;
    return 0;
}

protection_level_t medulla_get_protection_level(const medulla_t medulla) {
    if (!medulla) {
        return PROTECTION_LEVEL_NORMAL;
    }

    return medulla->protection_level;
}

circadian_phase_t medulla_get_circadian_phase(const medulla_t medulla) {
    if (!medulla) {
        return CIRCADIAN_PHASE_MORNING;
    }

    return medulla->circadian_phase;
}

int medulla_get_stats(const medulla_t medulla, medulla_stats_t* stats) {
    if (!medulla || !stats) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    *stats = medulla->stats;
    nimcp_platform_mutex_unlock(medulla->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int medulla_connect_bio_async(medulla_t medulla) {
    if (!medulla) {
        return -1;
    }

    if (medulla->bio_async_enabled) {
        return 0;  // Already connected
    }

    // TODO: Register with bio-async router when module ID is assigned
    // This will be implemented in Phase 1.2+ when bio-async module IDs are defined

    medulla->bio_async_enabled = false;  // Set to true when implemented

    NIMCP_LOGGING_INFO("Bio-async connection attempted (not yet implemented)");
    return 0;
}

int medulla_disconnect_bio_async(medulla_t medulla) {
    if (!medulla) {
        return -1;
    }

    if (!medulla->bio_async_enabled) {
        return 0;  // Not connected
    }

    // TODO: Unregister from bio-async router

    medulla->bio_async_enabled = false;
    medulla->bio_ctx = NULL;

    NIMCP_LOGGING_INFO("Bio-async disconnected");
    return 0;
}

bool medulla_is_bio_async_connected(const medulla_t medulla) {
    if (!medulla) {
        return false;
    }

    return medulla->bio_async_enabled;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* medulla_arousal_level_to_string(arousal_level_t level) {
    switch (level) {
        case AROUSAL_LEVEL_COMA: return "COMA";
        case AROUSAL_LEVEL_DEEP_SLEEP: return "DEEP_SLEEP";
        case AROUSAL_LEVEL_LIGHT_SLEEP: return "LIGHT_SLEEP";
        case AROUSAL_LEVEL_DROWSY: return "DROWSY";
        case AROUSAL_LEVEL_AWAKE: return "AWAKE";
        case AROUSAL_LEVEL_ALERT: return "ALERT";
        case AROUSAL_LEVEL_HYPERAROUSAL: return "HYPERAROUSAL";
        default: return "UNKNOWN";
    }
}

const char* medulla_protection_level_to_string(protection_level_t level) {
    switch (level) {
        case PROTECTION_LEVEL_NORMAL: return "NORMAL";
        case PROTECTION_LEVEL_CAUTIOUS: return "CAUTIOUS";
        case PROTECTION_LEVEL_GUARDED: return "GUARDED";
        case PROTECTION_LEVEL_DEFENSIVE: return "DEFENSIVE";
        case PROTECTION_LEVEL_CRITICAL: return "CRITICAL";
        case PROTECTION_LEVEL_SHUTDOWN: return "SHUTDOWN";
        default: return "UNKNOWN";
    }
}

const char* medulla_circadian_phase_to_string(circadian_phase_t phase) {
    switch (phase) {
        case CIRCADIAN_PHASE_EARLY_MORNING: return "EARLY_MORNING";
        case CIRCADIAN_PHASE_MORNING: return "MORNING";
        case CIRCADIAN_PHASE_AFTERNOON: return "AFTERNOON";
        case CIRCADIAN_PHASE_EVENING: return "EVENING";
        case CIRCADIAN_PHASE_LATE_EVENING: return "LATE_EVENING";
        case CIRCADIAN_PHASE_NIGHT: return "NIGHT";
        case CIRCADIAN_PHASE_DEEP_NIGHT: return "DEEP_NIGHT";
        case CIRCADIAN_PHASE_PRE_DAWN: return "PRE_DAWN";
        default: return "UNKNOWN";
    }
}

const char* medulla_state_to_string(medulla_state_t state) {
    switch (state) {
        case MEDULLA_STATE_STOPPED: return "STOPPED";
        case MEDULLA_STATE_STARTING: return "STARTING";
        case MEDULLA_STATE_RUNNING: return "RUNNING";
        case MEDULLA_STATE_DEGRADED: return "DEGRADED";
        case MEDULLA_STATE_EMERGENCY: return "EMERGENCY";
        case MEDULLA_STATE_STOPPING: return "STOPPING";
        default: return "UNKNOWN";
    }
}

//=============================================================================
// Test Helper Functions
//=============================================================================

int medulla_test_set_arousal(medulla_t medulla, float level) {
    if (!medulla) {
        return -1;
    }

    /* Clamp level to valid range */
    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;

    nimcp_platform_mutex_lock(medulla->mutex);

    /* Set current arousal directly */
    medulla->current_arousal = level;

    /* Update arousal level enum based on level */
    if (level < 0.05f) {
        medulla->arousal_level = AROUSAL_LEVEL_COMA;
    } else if (level < 0.15f) {
        medulla->arousal_level = AROUSAL_LEVEL_DEEP_SLEEP;
    } else if (level < 0.30f) {
        medulla->arousal_level = AROUSAL_LEVEL_LIGHT_SLEEP;
    } else if (level < 0.45f) {
        medulla->arousal_level = AROUSAL_LEVEL_DROWSY;
    } else if (level < 0.65f) {
        medulla->arousal_level = AROUSAL_LEVEL_AWAKE;
    } else if (level < 0.85f) {
        medulla->arousal_level = AROUSAL_LEVEL_ALERT;
    } else {
        medulla->arousal_level = AROUSAL_LEVEL_HYPERAROUSAL;
    }

    nimcp_platform_mutex_unlock(medulla->mutex);
    return 0;
}

int medulla_test_set_protection(medulla_t medulla, protection_level_t level) {
    if (!medulla) {
        return -1;
    }

    if (level < PROTECTION_LEVEL_NORMAL || level > PROTECTION_LEVEL_SHUTDOWN) {
        return -1;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->protection_level = level;
    nimcp_platform_mutex_unlock(medulla->mutex);
    return 0;
}

int medulla_test_set_circadian(medulla_t medulla, circadian_phase_t phase) {
    if (!medulla) {
        return -1;
    }

    if (phase < CIRCADIAN_PHASE_EARLY_MORNING || phase > CIRCADIAN_PHASE_PRE_DAWN) {
        return -1;
    }

    /* Map phase to circadian time (middle of each phase's range) */
    float phase_time;
    switch (phase) {
        case CIRCADIAN_PHASE_EARLY_MORNING: phase_time = 7.5f; break;   /* [6,9) */
        case CIRCADIAN_PHASE_MORNING:       phase_time = 10.5f; break;  /* [9,12) */
        case CIRCADIAN_PHASE_AFTERNOON:     phase_time = 13.5f; break;  /* [12,15) */
        case CIRCADIAN_PHASE_EVENING:       phase_time = 16.5f; break;  /* [15,18) */
        case CIRCADIAN_PHASE_LATE_EVENING:  phase_time = 19.5f; break;  /* [18,21) */
        case CIRCADIAN_PHASE_NIGHT:         phase_time = 22.5f; break;  /* [21,24) */
        case CIRCADIAN_PHASE_DEEP_NIGHT:    phase_time = 1.5f; break;   /* [0,3) */
        case CIRCADIAN_PHASE_PRE_DAWN:      phase_time = 4.5f; break;   /* [3,6) */
        default: phase_time = 10.5f; break;
    }

    nimcp_platform_mutex_lock(medulla->mutex);
    medulla->circadian_phase = phase;
    medulla->current_circadian_time = phase_time;
    nimcp_platform_mutex_unlock(medulla->mutex);
    return 0;
}
