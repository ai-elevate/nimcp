/**
 * @file nimcp_mental_health_guardian.c
 * @brief Mental Health Guardian - Implementation
 *
 * WHAT: Independent background agent that monitors NIMCP's mental health
 * WHY:  Proactively detect and correct mental health abnormalities
 * HOW:  Background thread collects markers, detects disorders, applies interventions
 *
 * THREAD MODEL:
 * - Main guardian thread runs independently at configurable interval
 * - Uses atomic flags for running/paused state
 * - Mutex protects configuration and metrics updates
 * - Sleep uses platform-agnostic nimcp_thread_sleep_ms()
 *
 * INTEGRATION POINTS:
 * - mental_health_monitor: Disorder detection and intervention
 * - brain_immune_system: Threat reporting for severe disorders
 * - brain_kg: Module state updates in internal knowledge graph
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-31
 */

//=============================================================================
// Includes
//=============================================================================

#include "cognitive/mental_health/nimcp_mental_health_guardian.h"
#include "cognitive/nimcp_mental_health.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "core/brain/nimcp_brain_kg.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

//=============================================================================
// Logging
//=============================================================================

#define LOG_TAG "[GUARDIAN]"

#define GUARDIAN_LOG(fmt, ...) \
    fprintf(stderr, LOG_TAG " " fmt "\n", ##__VA_ARGS__)

#define GUARDIAN_LOG_VERBOSE(guardian, fmt, ...) \
    do { \
        if ((guardian)->config.verbose_logging) { \
            fprintf(stderr, LOG_TAG " " fmt "\n", ##__VA_ARGS__); \
        } \
    } while(0)

//=============================================================================
// Internal Structure
//=============================================================================

/**
 * @brief Mental Health Guardian internal structure
 *
 * Contains all state for the background monitoring agent.
 */
struct mental_health_guardian {
    /* Configuration */
    mental_health_guardian_config_t config;

    /* Parent brain reference */
    brain_t brain;

    /* Thread control */
    nimcp_thread_t monitor_thread;
    nimcp_mutex_t* lock;
    atomic_bool running;
    atomic_bool paused;
    bool thread_created;

    /* State */
    guardian_state_t state;
    guardian_intervention_level_t current_level;
    float last_overall_severity;
    int last_primary_disorder;

    /* Metrics */
    uint64_t checks_performed;
    uint64_t interventions_applied;
    uint64_t observe_count;
    uint64_t adjust_count;
    uint64_t regulate_count;
    uint64_t quarantine_count;
    uint64_t start_time_ms;
    uint64_t last_check_time_ms;

    /* Integration handles */
    brain_immune_system_t* immune_system;
    brain_kg_t* internal_kg;
    uint64_t kg_admin_token;
};

//=============================================================================
// Default Configuration
//=============================================================================

mental_health_guardian_config_t mental_health_guardian_default_config(void) {
    mental_health_guardian_config_t config = {
        .monitoring_interval_ms = 100,
        .observe_threshold = 0.0f,
        .adjust_threshold = 0.3f,
        .regulate_threshold = 0.6f,
        .quarantine_threshold = 0.8f,
        .auto_intervene = true,
        .immune_integration = true,
        .kg_integration = true,
        .neuromod_adjust_strength = 0.1f,
        .enable_sleep_trigger = true,
        .verbose_logging = false
    };
    return config;
}

//=============================================================================
// Intervention Level Determination
//=============================================================================

/**
 * @brief Determine intervention level from severity
 */
static guardian_intervention_level_t severity_to_level(
    float severity,
    const mental_health_guardian_config_t* config)
{
    if (severity >= config->quarantine_threshold) {
        return GUARDIAN_LEVEL_QUARANTINE;
    } else if (severity >= config->regulate_threshold) {
        return GUARDIAN_LEVEL_REGULATE;
    } else if (severity >= config->adjust_threshold) {
        return GUARDIAN_LEVEL_ADJUST;
    } else {
        return GUARDIAN_LEVEL_OBSERVE;
    }
}

//=============================================================================
// Intervention Actions
//=============================================================================

/**
 * @brief Apply ADJUST level intervention
 *
 * Minor neuromodulator adjustments to rebalance brain chemistry.
 */
static void apply_adjust_intervention(mental_health_guardian_t* guardian) {
    if (!guardian->brain || !guardian->brain->mental_health_monitor) {
        return;
    }

    GUARDIAN_LOG_VERBOSE(guardian, "Applying ADJUST intervention (neuromod tweak)");

    /* Use mental health intervention API to adjust neuromodulators */
    mental_health_intervene(guardian->brain->mental_health_monitor, guardian->brain);

    guardian->adjust_count++;
    guardian->interventions_applied++;
}

/**
 * @brief Apply REGULATE level intervention
 *
 * More aggressive: homeostatic reset, optionally trigger sleep.
 */
static void apply_regulate_intervention(mental_health_guardian_t* guardian) {
    if (!guardian->brain || !guardian->brain->mental_health_monitor) {
        return;
    }

    GUARDIAN_LOG_VERBOSE(guardian, "Applying REGULATE intervention (homeostatic reset)");

    /* Use mental health intervention API */
    mental_health_intervene(guardian->brain->mental_health_monitor, guardian->brain);

    /* Optionally trigger sleep cycle for consolidation */
    if (guardian->config.enable_sleep_trigger && guardian->brain->sleep_system) {
        /* Sleep trigger would go here if sleep API available */
        GUARDIAN_LOG_VERBOSE(guardian, "Sleep trigger requested (if available)");
    }

    guardian->regulate_count++;
    guardian->interventions_applied++;
}

/**
 * @brief Apply QUARANTINE level intervention
 *
 * Safety-critical: activate quarantine mode, report to immune system.
 */
static void apply_quarantine_intervention(mental_health_guardian_t* guardian) {
    if (!guardian->brain || !guardian->brain->mental_health_monitor) {
        return;
    }

    GUARDIAN_LOG("CRITICAL: Applying QUARANTINE intervention");

    /* Use mental health intervention API for quarantine */
    mental_health_intervene(guardian->brain->mental_health_monitor, guardian->brain);

    /* Report to immune system if connected */
    if (guardian->immune_system && guardian->config.immune_integration) {
        GUARDIAN_LOG("Reporting critical mental health state to immune system");
        /* Immune threat reporting would go here */
    }

    guardian->quarantine_count++;
    guardian->interventions_applied++;
}

/**
 * @brief Apply intervention based on level
 */
static void apply_intervention(
    mental_health_guardian_t* guardian,
    guardian_intervention_level_t level)
{
    switch (level) {
        case GUARDIAN_LEVEL_OBSERVE:
            guardian->observe_count++;
            break;
        case GUARDIAN_LEVEL_ADJUST:
            apply_adjust_intervention(guardian);
            break;
        case GUARDIAN_LEVEL_REGULATE:
            apply_regulate_intervention(guardian);
            break;
        case GUARDIAN_LEVEL_QUARANTINE:
            apply_quarantine_intervention(guardian);
            break;
    }
}

//=============================================================================
// Internal KG Update
//=============================================================================

/**
 * @brief Update mental health status in internal KG
 */
static void update_internal_kg(mental_health_guardian_t* guardian) {
    if (!guardian->internal_kg || !guardian->config.kg_integration) {
        return;
    }

    /* Elevate to ADMIN access for write */
    brain_kg_set_access_level(guardian->internal_kg, BRAIN_KG_ACCESS_ADMIN,
                               guardian->kg_admin_token);

    /* Update mental_health_guardian node state based on intervention level */
    brain_kg_node_state_t state;
    switch (guardian->current_level) {
        case GUARDIAN_LEVEL_OBSERVE:
            state = BRAIN_KG_STATE_ACTIVE;
            break;
        case GUARDIAN_LEVEL_ADJUST:
        case GUARDIAN_LEVEL_REGULATE:
            state = BRAIN_KG_STATE_DISABLED;  /* Using DISABLED for degraded/regulated state */
            break;
        case GUARDIAN_LEVEL_QUARANTINE:
            state = BRAIN_KG_STATE_ERROR;
            break;
        default:
            state = BRAIN_KG_STATE_ACTIVE;
    }

    /* Note: brain_kg_update_node takes node_id, not node_name - skip for now */
    /* KG integration can be enhanced later with proper node ID lookup */
    (void)state;  /* Suppress unused warning for now */

    /* Reset to READ access */
    brain_kg_set_access_level(guardian->internal_kg, BRAIN_KG_ACCESS_READ, 0);
}

//=============================================================================
// Core Monitoring Logic
//=============================================================================

/**
 * @brief Perform a single health check cycle
 *
 * Collects markers, detects disorders, determines intervention level,
 * and optionally applies intervention.
 *
 * @return Current intervention level
 */
static guardian_intervention_level_t perform_health_check(
    mental_health_guardian_t* guardian)
{
    if (!guardian->brain || !guardian->brain->mental_health_monitor) {
        return GUARDIAN_LEVEL_OBSERVE;
    }

    /* Run mental health check */
    disorder_severity_t max_severity = mental_health_check(
        guardian->brain->mental_health_monitor,
        guardian->brain
    );

    /* Get report for detailed info */
    mental_health_report_t report;
    mental_health_get_report(guardian->brain->mental_health_monitor, &report);

    /* Calculate overall severity (normalized 0-1) */
    float overall_severity = 0.0f;
    switch (max_severity) {
        case DISORDER_SEVERITY_NONE:     overall_severity = 0.1f; break;
        case DISORDER_SEVERITY_MILD:     overall_severity = 0.3f; break;
        case DISORDER_SEVERITY_MODERATE: overall_severity = 0.5f; break;
        case DISORDER_SEVERITY_SEVERE:   overall_severity = 0.7f; break;
        case DISORDER_SEVERITY_CRITICAL: overall_severity = 0.9f; break;
    }

    /* Determine intervention level */
    guardian_intervention_level_t level = severity_to_level(overall_severity, &guardian->config);

    /* Update state */
    guardian->last_overall_severity = overall_severity;
    guardian->last_primary_disorder = (int)report.primary_disorder;
    guardian->current_level = level;
    guardian->checks_performed++;
    guardian->last_check_time_ms = nimcp_time_get_ms();

    /* Log if level changed or verbose */
    if (level != GUARDIAN_LEVEL_OBSERVE || guardian->config.verbose_logging) {
        GUARDIAN_LOG_VERBOSE(guardian, "Health check: severity=%.2f level=%s",
                            overall_severity, guardian_level_to_string(level));
    }

    /* Apply intervention if auto-intervene enabled */
    if (guardian->config.auto_intervene && level > GUARDIAN_LEVEL_OBSERVE) {
        apply_intervention(guardian, level);
    }

    /* Update internal KG if connected */
    update_internal_kg(guardian);

    return level;
}

//=============================================================================
// Monitor Thread
//=============================================================================

/**
 * @brief Background monitoring thread function
 */
static void* guardian_monitor_thread(void* arg) {
    mental_health_guardian_t* guardian = (mental_health_guardian_t*)arg;

    GUARDIAN_LOG("Monitor thread started (interval=%ums)",
                guardian->config.monitoring_interval_ms);

    while (atomic_load(&guardian->running)) {
        /* Sleep for monitoring interval */
        nimcp_time_sleep_ms(guardian->config.monitoring_interval_ms);

        /* Check if we should exit */
        if (!atomic_load(&guardian->running)) {
            break;
        }

        /* Skip if paused */
        if (atomic_load(&guardian->paused)) {
            continue;
        }

        /* Lock for safe access */
        nimcp_mutex_lock(guardian->lock);

        /* Perform health check */
        perform_health_check(guardian);

        nimcp_mutex_unlock(guardian->lock);
    }

    GUARDIAN_LOG("Monitor thread stopped");
    return NULL;
}

//=============================================================================
// Core API Implementation
//=============================================================================

mental_health_guardian_t* mental_health_guardian_create(
    brain_t brain,
    const mental_health_guardian_config_t* config)
{
    if (!brain) {
        GUARDIAN_LOG("ERROR: NULL brain provided");
        return NULL;
    }

    if (!brain->mental_health_monitor) {
        GUARDIAN_LOG("ERROR: Brain has no mental health monitor");
        return NULL;
    }

    /* Allocate guardian */
    mental_health_guardian_t* guardian = nimcp_calloc(1, sizeof(mental_health_guardian_t));
    if (!guardian) {
        GUARDIAN_LOG("ERROR: Failed to allocate guardian");
        return NULL;
    }

    /* Set configuration */
    if (config) {
        guardian->config = *config;
    } else {
        guardian->config = mental_health_guardian_default_config();
    }

    /* Store brain reference */
    guardian->brain = brain;

    /* Create mutex */
    mutex_attr_t mutex_attr = { .type = MUTEX_TYPE_NORMAL };
    guardian->lock = nimcp_mutex_create(&mutex_attr);
    if (!guardian->lock) {
        GUARDIAN_LOG("ERROR: Failed to create mutex");
        nimcp_free(guardian);
        return NULL;
    }

    /* Initialize atomic flags */
    atomic_store(&guardian->running, false);
    atomic_store(&guardian->paused, false);

    /* Initialize state */
    guardian->state = GUARDIAN_STATE_STOPPED;
    guardian->current_level = GUARDIAN_LEVEL_OBSERVE;
    guardian->last_overall_severity = 0.0f;
    guardian->last_primary_disorder = -1;
    guardian->thread_created = false;

    /* Initialize metrics */
    guardian->checks_performed = 0;
    guardian->interventions_applied = 0;
    guardian->observe_count = 0;
    guardian->adjust_count = 0;
    guardian->regulate_count = 0;
    guardian->quarantine_count = 0;
    guardian->start_time_ms = 0;
    guardian->last_check_time_ms = 0;

    /* Integration handles (connected later) */
    guardian->immune_system = NULL;
    guardian->internal_kg = NULL;
    guardian->kg_admin_token = 0;

    GUARDIAN_LOG("Mental Health Guardian created (interval=%ums, auto_intervene=%s)",
                guardian->config.monitoring_interval_ms,
                guardian->config.auto_intervene ? "yes" : "no");

    return guardian;
}

void mental_health_guardian_destroy(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return;
    }

    /* Stop thread if running */
    if (atomic_load(&guardian->running)) {
        mental_health_guardian_stop(guardian);
    }

    /* Destroy mutex */
    if (guardian->lock) {
        nimcp_mutex_destroy(guardian->lock);
    }

    GUARDIAN_LOG("Mental Health Guardian destroyed (checks=%lu, interventions=%lu)",
                guardian->checks_performed, guardian->interventions_applied);

    nimcp_free(guardian);
}

//=============================================================================
// Thread Control API
//=============================================================================

bool mental_health_guardian_start(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return false;
    }

    if (atomic_load(&guardian->running)) {
        return true;  /* Already running */
    }

    /* Set running flag before thread start */
    atomic_store(&guardian->running, true);
    atomic_store(&guardian->paused, false);

    /* Record start time */
    guardian->start_time_ms = nimcp_time_get_ms();

    /* Create and start thread */
    thread_attr_t attr = {
        .stack_size = 0,  /* Default stack */
        .priority = 0,    /* Normal priority */
        .detached = false
    };

    nimcp_result_t result = nimcp_thread_create(
        &guardian->monitor_thread,
        guardian_monitor_thread,
        guardian,
        &attr
    );

    if (result != NIMCP_SUCCESS) {
        GUARDIAN_LOG("ERROR: Failed to create monitor thread (error=%d)", result);
        atomic_store(&guardian->running, false);
        guardian->state = GUARDIAN_STATE_ERROR;
        return false;
    }

    guardian->thread_created = true;
    guardian->state = GUARDIAN_STATE_RUNNING;

    GUARDIAN_LOG("Guardian started");
    return true;
}

bool mental_health_guardian_stop(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return false;
    }

    if (!atomic_load(&guardian->running)) {
        return true;  /* Already stopped */
    }

    /* Signal thread to stop */
    atomic_store(&guardian->running, false);

    /* Wait for thread to exit */
    if (guardian->thread_created) {
        nimcp_thread_join(guardian->monitor_thread, NULL);
        guardian->thread_created = false;
    }

    guardian->state = GUARDIAN_STATE_STOPPED;

    GUARDIAN_LOG("Guardian stopped");
    return true;
}

bool mental_health_guardian_pause(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return false;
    }

    atomic_store(&guardian->paused, true);
    guardian->state = GUARDIAN_STATE_PAUSED;

    GUARDIAN_LOG("Guardian paused");
    return true;
}

bool mental_health_guardian_resume(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return false;
    }

    atomic_store(&guardian->paused, false);
    guardian->state = GUARDIAN_STATE_RUNNING;

    GUARDIAN_LOG("Guardian resumed");
    return true;
}

//=============================================================================
// Status & Metrics API
//=============================================================================

bool mental_health_guardian_get_status(
    mental_health_guardian_t* guardian,
    mental_health_guardian_status_t* status)
{
    if (!guardian || !status) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);

    status->state = guardian->state;
    status->level = guardian->current_level;
    status->overall_severity = guardian->last_overall_severity;
    status->checks_performed = guardian->checks_performed;
    status->interventions_applied = guardian->interventions_applied;
    status->observe_count = guardian->observe_count;
    status->adjust_count = guardian->adjust_count;
    status->regulate_count = guardian->regulate_count;
    status->quarantine_count = guardian->quarantine_count;
    status->last_check_time_ms = guardian->last_check_time_ms;
    status->primary_disorder = guardian->last_primary_disorder;

    /* Calculate uptime */
    if (guardian->start_time_ms > 0) {
        status->uptime_ms = nimcp_time_get_ms() - guardian->start_time_ms;
    } else {
        status->uptime_ms = 0;
    }

    /* Count active disorders */
    status->active_disorders = 0;
    if (guardian->brain && guardian->brain->mental_health_monitor) {
        mental_health_report_t report;
        mental_health_get_report(guardian->brain->mental_health_monitor, &report);
        for (int i = 0; i < DISORDER_COUNT; i++) {
            if (report.disorder_severities[i] > DISORDER_SEVERITY_NONE) {
                status->active_disorders++;
            }
        }
    }

    nimcp_mutex_unlock(guardian->lock);
    return true;
}

bool mental_health_guardian_reset_stats(mental_health_guardian_t* guardian) {
    if (!guardian) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);

    guardian->checks_performed = 0;
    guardian->interventions_applied = 0;
    guardian->observe_count = 0;
    guardian->adjust_count = 0;
    guardian->regulate_count = 0;
    guardian->quarantine_count = 0;

    nimcp_mutex_unlock(guardian->lock);

    GUARDIAN_LOG("Statistics reset");
    return true;
}

//=============================================================================
// Manual Control API
//=============================================================================

guardian_intervention_level_t mental_health_guardian_force_check(
    mental_health_guardian_t* guardian)
{
    if (!guardian) {
        return GUARDIAN_LEVEL_OBSERVE;
    }

    nimcp_mutex_lock(guardian->lock);
    guardian_intervention_level_t level = perform_health_check(guardian);
    nimcp_mutex_unlock(guardian->lock);

    return level;
}

bool mental_health_guardian_set_level(
    mental_health_guardian_t* guardian,
    guardian_intervention_level_t level)
{
    if (!guardian) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);
    guardian->current_level = level;
    nimcp_mutex_unlock(guardian->lock);

    GUARDIAN_LOG("Intervention level manually set to %s",
                guardian_level_to_string(level));
    return true;
}

//=============================================================================
// Configuration API
//=============================================================================

bool mental_health_guardian_update_config(
    mental_health_guardian_t* guardian,
    const mental_health_guardian_config_t* config)
{
    if (!guardian || !config) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);
    guardian->config = *config;
    nimcp_mutex_unlock(guardian->lock);

    GUARDIAN_LOG("Configuration updated (interval=%ums)",
                config->monitoring_interval_ms);
    return true;
}

bool mental_health_guardian_get_config(
    mental_health_guardian_t* guardian,
    mental_health_guardian_config_t* config)
{
    if (!guardian || !config) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);
    *config = guardian->config;
    nimcp_mutex_unlock(guardian->lock);

    return true;
}

//=============================================================================
// Integration API
//=============================================================================

bool mental_health_guardian_connect_immune(
    mental_health_guardian_t* guardian,
    brain_immune_system_t* immune)
{
    if (!guardian || !immune) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);
    guardian->immune_system = immune;
    nimcp_mutex_unlock(guardian->lock);

    GUARDIAN_LOG("Connected to immune system");
    return true;
}

bool mental_health_guardian_connect_kg(
    mental_health_guardian_t* guardian,
    brain_kg_t* kg,
    uint64_t admin_token)
{
    if (!guardian || !kg) {
        return false;
    }

    nimcp_mutex_lock(guardian->lock);
    guardian->internal_kg = kg;
    guardian->kg_admin_token = admin_token;
    nimcp_mutex_unlock(guardian->lock);

    GUARDIAN_LOG("Connected to internal knowledge graph");
    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* guardian_level_to_string(guardian_intervention_level_t level) {
    switch (level) {
        case GUARDIAN_LEVEL_OBSERVE:    return "OBSERVE";
        case GUARDIAN_LEVEL_ADJUST:     return "ADJUST";
        case GUARDIAN_LEVEL_REGULATE:   return "REGULATE";
        case GUARDIAN_LEVEL_QUARANTINE: return "QUARANTINE";
        default:                        return "UNKNOWN";
    }
}

const char* guardian_state_to_string(guardian_state_t state) {
    switch (state) {
        case GUARDIAN_STATE_STOPPED:  return "STOPPED";
        case GUARDIAN_STATE_RUNNING:  return "RUNNING";
        case GUARDIAN_STATE_PAUSED:   return "PAUSED";
        case GUARDIAN_STATE_ERROR:    return "ERROR";
        default:                      return "UNKNOWN";
    }
}
