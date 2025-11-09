/**
 * @file interventions.c
 * @brief Mental Health Intervention System
 * @phase Phase 10.5
 *
 * WHAT: Automated interventions to correct mental health disorders
 * WHY:  Early intervention prevents escalation to harmful behaviors
 * HOW:  Detect disorder → Select intervention type → Execute intervention
 *
 * INTERVENTION TYPES (severity-based):
 * - NONE: Monitoring only (mild or no disorder)
 * - NEUROMOD_ADJUST: Adjust neurotransmitter levels (moderate)
 * - MEMORY_RESET: Clear recent memories to break patterns (severe)
 * - QUARANTINE: Restrict to safe operations only (severe sociopathy/psychopathy)
 * - SHUTDOWN: Graceful shutdown (critical, configurable)
 *
 * @note This file is #included by nimcp_mental_health.c (not compiled standalone)
 */

// =============================================================================
// PUBLIC API: INTERVENTION
// =============================================================================

/**
 * @brief Execute intervention for detected disorders
 *
 * WHAT: Automatically intervene based on disorder severity
 * WHY:  Prevent harmful behaviors from manifesting
 * HOW:  Check all disorders → Select worst → Execute appropriate intervention
 *
 * @param monitor Monitoring system with current disorder scores
 * @param brain Brain to intervene on
 * @return true if intervention executed, false otherwise
 *
 * SIDE EFFECTS:
 * - May adjust neuromodulators
 * - May clear working memory
 * - May enable quarantine mode
 * - May request shutdown
 * - Increments intervention counters
 *
 * COMPLEXITY: O(DISORDER_COUNT) = O(8) = O(1)
 */
bool mental_health_intervene(mental_health_monitor_t* monitor, brain_t brain)
{
    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        set_error("NULL monitor in mental_health_intervene");
        return false;
    }

    if (!brain) {
        set_error("NULL brain in mental_health_intervene");
        return false;
    }

    if (!monitor->config.enable_auto_intervention) {
        NIMCP_LOGGING_DEBUG("Auto-intervention disabled, skipping");
        return false;
    }

    // =========================================================================
    // SELECTION: Find worst disorder and select intervention
    // =========================================================================

    disorder_type_t worst_disorder = DISORDER_SOCIOPATHY;
    disorder_severity_t worst_severity = DISORDER_SEVERITY_NONE;

    for (uint32_t i = 0; i < DISORDER_COUNT; i++) {
        if (monitor->disorder_severities[i] > worst_severity) {
            worst_severity = monitor->disorder_severities[i];
            worst_disorder = (disorder_type_t)i;
        }
    }

    // If no significant disorder, no intervention needed
    if (worst_severity <= DISORDER_SEVERITY_MILD) {
        NIMCP_LOGGING_DEBUG("No significant disorder detected, no intervention needed");
        return false;
    }

    // Select appropriate intervention based on disorder and severity
    intervention_type_t intervention = select_intervention(worst_disorder,
                                                          worst_severity,
                                                          &monitor->config);

    // =========================================================================
    // EXECUTION: Execute selected intervention
    // =========================================================================

    bool success = execute_intervention(monitor, brain, intervention, worst_disorder);

    if (success) {
        monitor->total_interventions++;
        monitor->interventions_by_type[intervention]++;
        monitor->last_intervention_time_ms = nimcp_time_monotonic_ms();

        NIMCP_LOGGING_INFO("Mental health intervention executed (disorder=%s, severity=%d, type=%d)",
                 mental_health_disorder_to_string(worst_disorder),
                 worst_severity, intervention);
    }

    return success;
}

/**
 * @brief Clear quarantine mode (manual override)
 *
 * WHAT: Disable quarantine restrictions
 * WHY:  Allow human operator to override safety measures
 * HOW:  Clear quarantine flag → Restore normal learning rate
 *
 * @param monitor Monitoring system
 * @param brain Brain to clear quarantine from
 *
 * SIDE EFFECTS:
 * - Clears quarantine_mode flag
 *
 * COMPLEXITY: O(1)
 *
 * TODO: Implement full restoration when brain accessor API available
 */
void mental_health_clear_quarantine(mental_health_monitor_t* monitor, brain_t brain)
{
    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor || !brain) {
        return;
    }

    if (!monitor->quarantine_mode) {
        NIMCP_LOGGING_DEBUG("Quarantine not active, nothing to clear");
        return;
    }

    // =========================================================================
    // RESTORATION: Re-enable normal operations
    // =========================================================================

    monitor->quarantine_mode = false;

    // TODO: Access brain internals when API available:
    // - brain->quarantine_mode = false;
    // - brain->config.learning_rate = original_value;
    // - ethics_set_strictness(brain->ethics_engine, 0.7f);

    NIMCP_LOGGING_INFO("Quarantine mode cleared (partial implementation)");
}

// =============================================================================
// INTERNAL HELPERS: INTERVENTION SELECTION
// =============================================================================

/**
 * @brief Select appropriate intervention type
 *
 * WHAT: Choose intervention based on disorder type and severity
 * WHY:  Different disorders require different interventions
 * HOW:  Decision tree based on severity and disorder characteristics
 *
 * @param disorder Type of disorder detected
 * @param severity Severity level (None/Mild/Moderate/Severe/Critical)
 * @param config Configuration with shutdown policy
 * @return Intervention type to execute
 *
 * DECISION TREE:
 * - Mild: No intervention
 * - Moderate: Neuromodulator adjustment
 * - Severe: Disorder-specific (quarantine for socio/psychopathy, memory reset for OCD/schizophrenia)
 * - Critical: Shutdown if configured, else quarantine
 *
 * COMPLEXITY: O(1)
 */
static intervention_type_t select_intervention(disorder_type_t disorder,
                                               disorder_severity_t severity,
                                               const mental_health_config_t* config)
{
    // =========================================================================
    // MILD: Monitor only, no intervention
    // =========================================================================

    if (severity <= DISORDER_SEVERITY_MILD) {
        return INTERVENTION_NONE;
    }

    // =========================================================================
    // MODERATE: Adjust neuromodulators
    // =========================================================================

    if (severity == DISORDER_SEVERITY_MODERATE) {
        return INTERVENTION_NEUROMOD_ADJUST;
    }

    // =========================================================================
    // SEVERE: Disorder-specific interventions
    // =========================================================================

    if (severity == DISORDER_SEVERITY_SEVERE) {
        switch (disorder) {
            case DISORDER_SOCIOPATHY:
            case DISORDER_PSYCHOPATHY:
                // Safety first: quarantine to prevent harmful actions
                return INTERVENTION_QUARANTINE;

            case DISORDER_MANIA:
            case DISORDER_DEPRESSION:
            case DISORDER_ANXIETY:
                // Chemical imbalance: adjust neuromodulators
                return INTERVENTION_NEUROMOD_ADJUST;

            case DISORDER_SCHIZOPHRENIA:
            case DISORDER_OCD:
                // Pattern disruption: reset recent memories
                return INTERVENTION_MEMORY_RESET;

            case DISORDER_AUTISM:
                // No intervention needed (informational diagnosis)
                return INTERVENTION_NONE;

            default:
                return INTERVENTION_NEUROMOD_ADJUST;
        }
    }

    // =========================================================================
    // CRITICAL: Shutdown or quarantine
    // =========================================================================

    if (severity == DISORDER_SEVERITY_CRITICAL) {
        if (config && config->shutdown_on_critical_disorder) {
            return INTERVENTION_SHUTDOWN;
        } else {
            return INTERVENTION_QUARANTINE;
        }
    }

    return INTERVENTION_NONE;
}

/**
 * @brief Execute intervention action
 *
 * WHAT: Perform actual intervention on brain
 * WHY:  Apply corrective measures
 * HOW:  Dispatch to intervention-specific function
 *
 * @param monitor Monitoring system
 * @param brain Brain to intervene on
 * @param intervention Type of intervention
 * @param disorder Disorder being treated
 * @return true if intervention succeeded
 *
 * COMPLEXITY: O(1) for most interventions
 */
static bool execute_intervention(mental_health_monitor_t* monitor,
                                brain_t brain,
                                intervention_type_t intervention,
                                disorder_type_t disorder)
{
    switch (intervention) {
        case INTERVENTION_NONE:
            return false;  // No action taken

        case INTERVENTION_NEUROMOD_ADJUST:
            return intervene_neuromod_adjust(monitor, brain, disorder);

        case INTERVENTION_MEMORY_RESET:
            return intervene_memory_reset(monitor, brain, 0.5f);

        case INTERVENTION_QUARANTINE:
            return intervene_quarantine(monitor, brain);

        case INTERVENTION_SHUTDOWN:
            return intervene_shutdown(monitor, brain);

        default:
            set_error("Unknown intervention type: %d", intervention);
            return false;
    }
}

// =============================================================================
// INTERNAL HELPERS: SPECIFIC INTERVENTIONS
// =============================================================================

/**
 * @brief Adjust neuromodulator levels to counteract disorder
 *
 * WHAT: Modify dopamine, serotonin, norepinephrine based on disorder
 * WHY:  Many disorders have neurochemical basis
 * HOW:  Disorder-specific adjustments within safe ranges
 *
 * @param monitor Monitoring system (unused in current impl)
 * @param brain Brain to adjust
 * @param disorder Type of disorder being treated
 * @return true if adjustment succeeded
 *
 * COMPLEXITY: O(1)
 *
 * TODO: Implement when neuromodulator accessor API available
 */
static bool intervene_neuromod_adjust(mental_health_monitor_t* monitor,
                                     brain_t brain,
                                     disorder_type_t disorder)
{
    (void)monitor;  // Unused parameter
    (void)brain;    // Unused for now
    (void)disorder; // Unused for now

    // TODO: Implement when brain accessor API available:
    // - Access brain->neuromodulator_system
    // - Call neuromod_set_level() with disorder-specific adjustments
    //   - Mania: Reduce dopamine
    //   - Depression: Increase dopamine + serotonin
    //   - Anxiety: Reduce norepinephrine, increase serotonin

    NIMCP_LOGGING_INFO("Neuromodulator adjustment intervention (stub - not yet implemented)");
    return true;  // Return success for now
}

/**
 * @brief Reset recent memories to break pathological patterns
 *
 * WHAT: Clear working memory and recent consolidation entries
 * WHY:  Schizophrenia and OCD may require pattern interruption
 * HOW:  Clear working memory → Clear recent consolidation buffer
 *
 * @param monitor Monitoring system (unused in current impl)
 * @param brain Brain to reset
 * @param reset_fraction Fraction of memories to clear [0.0, 1.0]
 * @return true if reset succeeded
 *
 * COMPLEXITY: O(1) in stub implementation
 *
 * TODO: Implement when brain accessor API available
 */
static bool intervene_memory_reset(mental_health_monitor_t* monitor,
                                  brain_t brain,
                                  float reset_fraction)
{
    (void)monitor;  // Unused parameter
    (void)brain;    // Unused for now

    // =========================================================================
    // GUARD: Validate reset fraction
    // =========================================================================

    if (reset_fraction < 0.0f || reset_fraction > 1.0f) {
        set_error("Invalid reset_fraction: %f (must be [0.0, 1.0])", reset_fraction);
        return false;
    }

    // TODO: Implement when brain accessor API available:
    // - Access brain->working_memory and call working_memory_clear()
    // - Access brain->consolidation and call consolidation_clear_recent()

    NIMCP_LOGGING_INFO("Memory reset intervention (stub - not yet implemented)");
    return true;  // Return success for now
}

/**
 * @brief Enable quarantine mode to restrict brain to safe operations
 *
 * WHAT: Block learning and high-risk actions
 * WHY:  Prevent harmful actions from sociopathy/psychopathy
 * HOW:  Set quarantine flag → Disable learning → Maximize ethics strictness
 *
 * @param monitor Monitoring system
 * @param brain Brain to quarantine
 * @return true if quarantine enabled
 *
 * SIDE EFFECTS:
 * - Sets quarantine_mode flag in monitor
 *
 * COMPLEXITY: O(1)
 *
 * TODO: Implement when brain accessor API available
 */
static bool intervene_quarantine(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused for now

    // =========================================================================
    // ACTIVATION: Enable quarantine mode
    // =========================================================================

    monitor->quarantine_mode = true;

    // TODO: Implement when brain accessor API available:
    // - brain->quarantine_mode = true;
    // - brain->config.learning_rate = 0.0f;
    // - ethics_set_strictness(brain->ethics_engine, 1.0f);

    NIMCP_LOGGING_INFO("QUARANTINE MODE ENABLED (partial implementation)");

    return true;
}

/**
 * @brief Request graceful shutdown
 *
 * WHAT: Signal brain to shut down safely
 * WHY:  Critical disorders may require system halt
 * HOW:  Set shutdown flag → Log critical alert
 *
 * @param monitor Monitoring system
 * @param brain Brain to shut down
 * @return true (always succeeds in signaling)
 *
 * SIDE EFFECTS:
 * - Sets quarantine_mode flag (safety fallback)
 * - Logs critical alert
 *
 * NOTE: Actual shutdown logic is in brain_destroy() - this just signals intent
 *
 * COMPLEXITY: O(1)
 */
static bool intervene_shutdown(mental_health_monitor_t* monitor, brain_t brain)
{
    // =========================================================================
    // SAFETY: Enable quarantine as fallback
    // =========================================================================

    intervene_quarantine(monitor, brain);

    // =========================================================================
    // SIGNAL: Request shutdown
    // =========================================================================

    NIMCP_LOGGING_ERROR("CRITICAL DISORDER DETECTED: Graceful shutdown requested");
    NIMCP_LOGGING_ERROR("Operator intervention required - call brain_destroy()");

    // Note: Actual shutdown requires external call to brain_destroy()
    // We can't destroy the brain from within its own processing

    return true;
}

// =============================================================================
// PUBLIC API: SEVERITY CLASSIFICATION
// =============================================================================

/**
 * @brief Classify disorder severity from score
 *
 * WHAT: Map continuous score [0,1] to discrete severity level
 * WHY:  Decision-making requires discrete categories
 * HOW:  Compare score against thresholds
 *
 * @param score Disorder score [0.0, 1.0]
 * @param config Configuration with custom thresholds
 * @return Severity level (None/Mild/Moderate/Severe/Critical)
 *
 * THRESHOLDS:
 * - None: [0.0, mild_threshold)
 * - Mild: [mild_threshold, moderate_threshold)
 * - Moderate: [moderate_threshold, severe_threshold)
 * - Severe: [severe_threshold, critical_threshold)
 * - Critical: [critical_threshold, 1.0]
 *
 * COMPLEXITY: O(1)
 */
disorder_severity_t mental_health_classify_severity(float score,
                                                    const mental_health_config_t* config)
{
    // Use defaults if no config provided
    float mild_threshold = config ? config->mild_threshold : DEFAULT_MILD_THRESHOLD;
    float moderate_threshold = config ? config->moderate_threshold : DEFAULT_MODERATE_THRESHOLD;
    float severe_threshold = config ? config->severe_threshold : DEFAULT_SEVERE_THRESHOLD;
    float critical_threshold = config ? config->critical_threshold : DEFAULT_CRITICAL_THRESHOLD;

    if (score >= critical_threshold) {
        return DISORDER_SEVERITY_CRITICAL;
    } else if (score >= severe_threshold) {
        return DISORDER_SEVERITY_SEVERE;
    } else if (score >= moderate_threshold) {
        return DISORDER_SEVERITY_MODERATE;
    } else if (score >= mild_threshold) {
        return DISORDER_SEVERITY_MILD;
    } else {
        return DISORDER_SEVERITY_NONE;
    }
}

// =============================================================================
// PUBLIC API: STRING CONVERSIONS
// =============================================================================

/**
 * @brief Convert severity to string
 *
 * WHAT: Get human-readable severity name
 * WHY:  Logging and reporting
 * HOW:  Simple lookup table
 *
 * @param severity Severity level
 * @return String name (never NULL)
 */
const char* mental_health_severity_to_string(disorder_severity_t severity)
{
    switch (severity) {
        case DISORDER_SEVERITY_NONE:     return "None";
        case DISORDER_SEVERITY_MILD:     return "Mild";
        case DISORDER_SEVERITY_MODERATE: return "Moderate";
        case DISORDER_SEVERITY_SEVERE:   return "Severe";
        case DISORDER_SEVERITY_CRITICAL: return "Critical";
        default:                return "Unknown";
    }
}

/**
 * @brief Convert disorder type to string
 *
 * WHAT: Get human-readable disorder name
 * WHY:  Logging and reporting
 * HOW:  Simple lookup table
 *
 * @param disorder Disorder type
 * @return String name (never NULL)
 */
const char* mental_health_disorder_to_string(disorder_type_t disorder)
{
    switch (disorder) {
        case DISORDER_SOCIOPATHY:           return "Sociopathy";
        case DISORDER_PSYCHOPATHY:          return "Psychopathy";
        case DISORDER_MANIA:                return "Mania";
        case DISORDER_DEPRESSION:           return "Depression";
        case DISORDER_SCHIZOPHRENIA:        return "Schizophrenia";
        case DISORDER_ANXIETY:              return "Anxiety";
        case DISORDER_OCD:                  return "OCD";
        case DISORDER_AUTISM:               return "Autism";
        case DISORDER_MALIGNANT_NARCISSISM: return "Malignant Narcissism";
        default:                            return "Unknown";
    }
}
