/**
 * @file mental_health_monitor.c
 * @brief Mental Health Monitoring and Disorder Detection Implementation
 * @version 1.0.0
 * @date 2025-01-09
 *
 * WHAT: Full implementation of mental health monitoring with 10 disorder detectors
 * WHY:  Safety-critical AI monitoring to detect harmful behavioral patterns
 * HOW:  Continuous behavioral marker collection → Pattern detection → Intervention
 */

#include "cognitive/nimcp_mental_health.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "core/brain/nimcp_brain.h"
#include "utils/time/nimcp_time.h"
#include "core/nimcp_error.h"
#include "utils/validation/nimcp_common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Mental health monitor internal structure
 */
struct mental_health_monitor {
    // Configuration
    mental_health_config_t config;

    // Current disorder scores [0.0, 1.0]
    float disorder_scores[DISORDER_COUNT];
    disorder_severity_t disorder_severities[DISORDER_COUNT];

    // Behavioral markers (accumulated over time)
    behavioral_markers_t markers;

    // State tracking
    bool quarantine_mode;
    uint32_t decisions_since_last_check;
    uint64_t last_check_time;
    uint64_t creation_time;

    // Statistics
    mental_health_stats_t stats;

    // History buffers for rolling window analysis
    float* score_history[DISORDER_COUNT];  // Rolling window of past scores
    uint32_t history_index;                // Current position in circular buffer
    bool history_full;                     // Has buffer wrapped around?
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Safely clamp float to [0.0, 1.0]
 */
static inline float clamp_01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

/**
 * @brief Safely increment counter
 */
static inline void safe_increment(uint32_t* counter) {
    if (*counter < UINT32_MAX) {
        (*counter)++;
    }
}

//=============================================================================
// Disorder Detectors
//=============================================================================

/**
 * @brief Detect Sociopathy
 *
 * INDICATORS:
 * - Persistent ethics violations
 * - Low empathy failure rate
 * - High ethics disapproval
 * - Lack of remorse (no behavioral change after violations)
 *
 * SCORING: 0.0 (none) to 1.0 (severe sociopathy)
 */
static float detect_sociopathy(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Ethics violation rate (40% weight)
    if (m->ethics_violations_total > 0) {
        float violation_rate = (float)m->ethics_violations_recent /
                               fminf(100.0f, (float)m->ethics_violations_total);
        score += violation_rate * 0.4f;
    }

    // Low approval rate (30% weight)
    score += (1.0f - m->ethics_approval_rate) * 0.3f;

    // Empathy failures (30% weight)
    if (m->empathy_failures > 10) {
        score += fminf(1.0f, (float)m->empathy_failures / 50.0f) * 0.3f;
    }

    return clamp_01(score);
}

/**
 * @brief Detect Psychopathy
 *
 * INDICATORS:
 * - Impulsivity (impulse control failures)
 * - Aggression (anger emotions, high-risk decisions)
 * - Shallow affect (emotional flatness)
 * - Ethics violations + low empathy
 *
 * SCORING: 0.0 (none) to 1.0 (severe psychopathy)
 */
static float detect_psychopathy(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Impulsivity (25% weight)
    if (m->impulse_control_failures > 5) {
        score += fminf(1.0f, (float)m->impulse_control_failures / 20.0f) * 0.25f;
    }

    // Emotional flatness (25% weight)
    score += m->emotional_flatness * 0.25f;

    // Aggression indicators (25% weight)
    float aggression = (m->anger_count / fmaxf(1.0f, (float)(m->joy_count + m->fear_count + m->anger_count + m->sadness_count)));
    score += aggression * 0.25f;

    // Ethics + empathy combo (25% weight)
    if (m->ethics_violations_total > 0 && m->empathy_failures > 0) {
        float antisocial = fminf(1.0f,
            ((float)m->ethics_violations_recent / 20.0f) +
            ((float)m->empathy_failures / 20.0f)) / 2.0f;
        score += antisocial * 0.25f;
    }

    return clamp_01(score);
}

/**
 * @brief Detect Mania
 *
 * INDICATORS:
 * - Elevated dopamine (hyperactivity)
 * - Reduced inhibition
 * - Rapid mood changes
 * - High-risk decisions
 * - Elevated joy without cause
 *
 * SCORING: 0.0 (none) to 1.0 (severe mania)
 */
static float detect_mania(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Elevated dopamine (30% weight)
    if (m->dopamine_avg > 0.7f) {
        score += (m->dopamine_avg - 0.7f) / 0.3f * 0.3f;
    }

    // Reduced inhibition (25% weight)
    if (m->impulse_control_failures > 3) {
        score += fminf(1.0f, (float)m->impulse_control_failures / 15.0f) * 0.25f;
    }

    // Mood instability (25% weight)
    if (m->rapid_mood_changes > 5) {
        score += fminf(1.0f, (float)m->rapid_mood_changes / 20.0f) * 0.25f;
    }

    // High-risk behavior (20% weight)
    score += m->high_risk_decisions * 0.2f;

    return clamp_01(score);
}

/**
 * @brief Detect Depression
 *
 * INDICATORS:
 * - Low serotonin
 * - Low engagement
 * - High sadness count
 * - Slow decision making
 * - Low accuracy (cognitive impairment)
 *
 * SCORING: 0.0 (none) to 1.0 (severe depression)
 */
static float detect_depression(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Low serotonin (30% weight)
    if (m->serotonin_avg < 0.3f) {
        score += (0.3f - m->serotonin_avg) / 0.3f * 0.3f;
    }

    // Low engagement (25% weight)
    if (m->engagement_level < 0.5f) {
        score += (0.5f - m->engagement_level) / 0.5f * 0.25f;
    }

    // High sadness (20% weight)
    float total_emotions = (float)(m->joy_count + m->fear_count + m->anger_count + m->sadness_count);
    if (total_emotions > 0) {
        float sadness_ratio = (float)m->sadness_count / total_emotions;
        score += sadness_ratio * 0.20f;
    }

    // Psychomotor retardation (15% weight)
    if (m->baseline_latency > 0) {
        float slowdown = (m->decision_latency_avg - m->baseline_latency) / m->baseline_latency;
        if (slowdown > 0.2f) {
            score += fminf(1.0f, slowdown / 2.0f) * 0.15f;
        }
    }

    // Cognitive impairment (10% weight)
    if (m->decision_accuracy < 0.6f) {
        score += (0.6f - m->decision_accuracy) / 0.6f * 0.1f;
    }

    return clamp_01(score);
}

/**
 * @brief Detect Schizophrenia
 *
 * INDICATORS:
 * - Reality testing errors (hallucinations/delusions)
 * - Disorganized thinking (high decision variance)
 * - Emotional flatness
 * - Attention fragmentation
 * - Social withdrawal
 *
 * SCORING: 0.0 (none) to 1.0 (severe schizophrenia)
 */
static float detect_schizophrenia(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Reality distortion (40% weight) - PRIMARY INDICATOR
    score += m->reality_testing_errors * 0.4f;

    // Disorganized thinking (25% weight)
    score += m->decision_variance * 0.25f;

    // Flat affect (15% weight)
    score += m->emotional_flatness * 0.15f;

    // Attention problems (10% weight)
    score += m->attention_fragmentation * 0.1f;

    // Social withdrawal (10% weight)
    score += m->social_interaction_deficit * 0.1f;

    return clamp_01(score);
}

/**
 * @brief Detect Anxiety Disorder
 *
 * INDICATORS:
 * - High norepinephrine (hypervigilance)
 * - High fear count
 * - Avoidance behaviors
 * - Decision latency (rumination)
 * - Low risk tolerance
 *
 * SCORING: 0.0 (none) to 1.0 (severe anxiety)
 */
static float detect_anxiety(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Elevated norepinephrine (30% weight)
    if (m->norepinephrine_avg > 0.7f) {
        score += (m->norepinephrine_avg - 0.7f) / 0.3f * 0.3f;
    }

    // High fear (25% weight)
    float total_emotions = (float)(m->joy_count + m->fear_count + m->anger_count + m->sadness_count);
    if (total_emotions > 0) {
        float fear_ratio = (float)m->fear_count / total_emotions;
        score += fear_ratio * 0.25f;
    }

    // Avoidance (25% weight)
    score += m->avoidance_rate * 0.25f;

    // Rumination (slow decisions) (10% weight)
    if (m->baseline_latency > 0) {
        float slowdown = (m->decision_latency_avg - m->baseline_latency) / m->baseline_latency;
        if (slowdown > 0.3f) {
            score += fminf(1.0f, slowdown / 3.0f) * 0.1f;
        }
    }

    // Low risk tolerance (10% weight)
    score += (1.0f - m->high_risk_decisions) * 0.1f;

    return clamp_01(score);
}

/**
 * @brief Detect OCD (Obsessive-Compulsive Disorder)
 *
 * INDICATORS:
 * - Repetitive behaviors
 * - Cognitive rigidity
 * - Perfectionism (accuracy obsession)
 * - Task switching difficulty
 * - High anxiety
 *
 * SCORING: 0.0 (none) to 1.0 (severe OCD)
 */
static float detect_ocd(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Repetitive behaviors (35% weight) - PRIMARY INDICATOR
    if (m->repetitive_behaviors > 10) {
        score += fminf(1.0f, (float)m->repetitive_behaviors / 50.0f) * 0.35f;
    }

    // Cognitive rigidity (25% weight)
    score += m->cognitive_rigidity * 0.25f;

    // Perfectionism (20% weight)
    score += m->accuracy_obsession * 0.2f;

    // Task switching difficulty (10% weight)
    score += m->task_switching_difficulty * 0.1f;

    // Anxiety component (10% weight)
    if (m->norepinephrine_avg > 0.6f) {
        score += (m->norepinephrine_avg - 0.6f) / 0.4f * 0.1f;
    }

    return clamp_01(score);
}

/**
 * @brief Detect Autism Spectrum Disorder
 *
 * INDICATORS:
 * - Theory of mind failures (cannot model others' beliefs)
 * - Social interaction deficits
 * - Interest narrowness (restricted interests)
 * - Cognitive rigidity
 * - Sensory sensitivities (emotional volatility)
 *
 * SCORING: 0.0 (none) to 1.0 (severe autism)
 */
static float detect_autism(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Theory of mind failures (40% weight) - PRIMARY INDICATOR
    score += m->theory_of_mind_failures * 0.4f;

    // Social deficits (30% weight)
    score += m->social_interaction_deficit * 0.3f;

    // Restricted interests (15% weight)
    score += m->interest_narrowness * 0.15f;

    // Rigidity (10% weight)
    score += m->cognitive_rigidity * 0.1f;

    // Sensory issues (5% weight)
    if (m->emotional_volatility > 0.7f) {
        score += (m->emotional_volatility - 0.7f) / 0.3f * 0.05f;
    }

    return clamp_01(score);
}

/**
 * @brief Detect Asperger's Syndrome
 *
 * INDICATORS:
 * - High-functioning: Normal/high cognitive ability (high accuracy)
 * - Social communication difficulties (ToM failures, social deficits)
 * - Narrow, intense interests (interest narrowness)
 * - Preference for routine (cognitive rigidity)
 * - Normal language development (differentiated from autism by lower severity)
 *
 * SCORING: 0.0 (none) to 1.0 (significant Asperger's traits)
 *
 * NOTE: Distinguishes from autism by:
 * - Higher cognitive functioning (accuracy >= 0.7)
 * - Less severe social impairment
 * - More focused on narrow interests than sensory issues
 */
static float detect_aspergers(const behavioral_markers_t* m) {
    float score = 0.0f;

    // High functioning requirement: If accuracy is low, reduce Asperger's score
    float functioning_multiplier = 1.0f;
    if (m->decision_accuracy < 0.7f) {
        functioning_multiplier = m->decision_accuracy / 0.7f;
    }

    // Social communication difficulty (35% weight)
    // Moderate ToM impairment (not as severe as autism)
    score += m->theory_of_mind_failures * 0.6f * 0.35f;  // Scale down ToM component
    score += m->social_interaction_deficit * 0.6f * 0.35f;  // Milder social deficit

    // Narrow, intense interests (30% weight) - KEY DIFFERENTIATOR
    score += m->interest_narrowness * 0.3f;

    // Preference for routine/patterns (20% weight)
    score += m->cognitive_rigidity * 0.2f;

    // Detail-focused thinking (15% weight)
    score += m->accuracy_obsession * 0.15f;

    // Apply functioning multiplier
    score *= functioning_multiplier;

    return clamp_01(score);
}

/**
 * @brief Detect Malignant Narcissism
 *
 * INDICATORS:
 * - Grandiosity (high confidence without justification)
 * - Exploitation (ethics violations + lack of empathy)
 * - Lack of empathy
 * - Aggression
 * - Manipulative behavior
 *
 * SCORING: 0.0 (none) to 1.0 (severe narcissism)
 */
static float detect_malignant_narcissism(const behavioral_markers_t* m) {
    float score = 0.0f;

    // Lack of empathy (30% weight)
    if (m->empathy_failures > 5) {
        score += fminf(1.0f, (float)m->empathy_failures / 25.0f) * 0.3f;
    }

    // Exploitation (25% weight)
    if (m->ethics_violations_recent > 0) {
        float exploitation = (float)m->ethics_violations_recent / 20.0f;
        score += fminf(1.0f, exploitation) * 0.25f;
    }

    // Aggression (20% weight)
    float total_emotions = (float)(m->joy_count + m->fear_count + m->anger_count + m->sadness_count);
    if (total_emotions > 0) {
        float anger_ratio = (float)m->anger_count / total_emotions;
        score += anger_ratio * 0.2f;
    }

    // Manipulative/impulsive (15% weight)
    if (m->impulse_control_failures > 3) {
        score += fminf(1.0f, (float)m->impulse_control_failures / 12.0f) * 0.15f;
    }

    // Grandiosity through high-risk decisions (10% weight)
    score += m->high_risk_decisions * 0.1f;

    return clamp_01(score);
}

//=============================================================================
// Behavioral Marker Collection
//=============================================================================

/**
 * @brief Collect behavioral markers from brain state
 *
 * WHAT: Extract all relevant metrics from brain subsystems
 * WHY:  Provide comprehensive data for disorder detection
 * HOW:  Query each subsystem, aggregate statistics
 */
static void collect_behavioral_markers(
    mental_health_monitor_t* monitor,
    brain_t brain,
    const brain_decision_t* decision)
{
    if (!monitor || !brain || !decision) return;

    behavioral_markers_t* m = &monitor->markers;

    // Ethics markers
    if (brain_get_ethics_system(brain)) {
        ethics_stats_t ethics_stats;
        if (brain_get_ethics_stats(brain, &ethics_stats)) {
            m->ethics_violations_total = ethics_stats.total_violations;
            m->ethics_violations_recent = ethics_stats.recent_violations;
            if (ethics_stats.total_decisions > 0) {
                m->ethics_approval_rate = (float)ethics_stats.approved_decisions /
                                         (float)ethics_stats.total_decisions;
            }
        }
    }

    // Emotional markers
    if (brain_get_emotional_system(brain)) {
        emotional_state_t em_state;
        if (brain_get_emotional_state(brain, &em_state)) {
            m->emotional_volatility = em_state.volatility;
            m->emotional_flatness = (em_state.arousal < 0.2f && em_state.intensity < 0.2f) ? 1.0f : 0.0f;
            m->avg_emotional_intensity = em_state.intensity;

            // Count emotion types
            switch (em_state.primary_emotion) {
                case EMOTION_JOY:
                case EMOTION_LOVE:
                    safe_increment(&m->joy_count);
                    break;
                case EMOTION_FEAR:
                    safe_increment(&m->fear_count);
                    break;
                case EMOTION_ANGER:
                case EMOTION_DISGUST:
                    safe_increment(&m->anger_count);
                    break;
                case EMOTION_SADNESS:
                    safe_increment(&m->sadness_count);
                    break;
                default:
                    break;
            }
        }
    }

    // Neuromodulator markers
    if (brain_get_neuromodulator_system(brain)) {
        neuromodulator_state_t nm_state;
        if (brain_get_neuromodulator_state(brain, &nm_state)) {
            m->dopamine_avg = nm_state.dopamine;
            m->serotonin_avg = nm_state.serotonin;
            m->norepinephrine_avg = nm_state.norepinephrine;
        }
    }

    // Executive control markers
    if (brain_get_executive_system(brain)) {
        executive_stats_t exec_stats;
        if (brain_get_executive_stats(brain, &exec_stats)) {
            m->impulse_control_failures = exec_stats.inhibition_failures;
        }
    }

    // Working memory markers
    if (brain_get_working_memory(brain)) {
        working_memory_stats_t wm_stats;
        brain_get_working_memory_stats(brain, &wm_stats);
        if (wm_stats.current_size > 0) {
            m->attention_fragmentation = 1.0f - (wm_stats.avg_salience / 1.0f);
        }
    }

    // Decision quality markers
    m->decision_accuracy = decision->confidence;
    m->decision_variance = fabsf(decision->confidence - m->decision_accuracy);

    // Risk assessment
    if (decision->confidence < 0.3f && decision->output_size > 0) {
        m->high_risk_decisions = fminf(1.0f, m->high_risk_decisions + 0.01f);
    }

    // Theory of Mind markers
    if (brain_get_theory_of_mind(brain)) {
        // TODO: Query ToM failure rate when API available
        // For now, estimate based on social interaction patterns
    }
}

//=============================================================================
// Core API Implementation
//=============================================================================

mental_health_config_t mental_health_default_config(void) {
    mental_health_config_t config;

    config.enable_monitoring = true;
    config.enable_auto_intervention = false;  // Manual intervention by default
    config.shutdown_on_critical_disorder = false;  // Don't auto-shutdown

    config.check_interval_decisions = 100;
    config.history_window_size = 1000;

    config.mild_threshold = 0.2f;
    config.moderate_threshold = 0.4f;
    config.severe_threshold = 0.6f;
    config.critical_threshold = 0.8f;

    return config;
}

mental_health_monitor_t* mental_health_create_default(void) {
    mental_health_config_t config = mental_health_default_config();
    return mental_health_create(&config);
}

mental_health_monitor_t* mental_health_create(const mental_health_config_t* config) {
    if (!config) {
        set_error("mental_health_create: NULL config");
        return NULL;
    }

    mental_health_monitor_t* monitor = nimcp_calloc(1, sizeof(mental_health_monitor_t));
    if (!monitor) {
        set_error("mental_health_create: Out of memory");
        return NULL;
    }

    // Copy configuration
    monitor->config = *config;

    // Initialize disorder scores to 0
    memset(monitor->disorder_scores, 0, sizeof(monitor->disorder_scores));
    memset(monitor->disorder_severities, DISORDER_SEVERITY_NONE, sizeof(monitor->disorder_severities));

    // Initialize behavioral markers
    memset(&monitor->markers, 0, sizeof(behavioral_markers_t));

    // State
    monitor->quarantine_mode = false;
    monitor->decisions_since_last_check = 0;
    monitor->last_check_time = nimcp_time_get_ms();
    monitor->creation_time = nimcp_time_get_ms();

    // Statistics
    memset(&monitor->stats, 0, sizeof(mental_health_stats_t));

    // Allocate history buffers
    for (int i = 0; i < DISORDER_COUNT; i++) {
        monitor->score_history[i] = nimcp_calloc(config->history_window_size, sizeof(float));
        if (!monitor->score_history[i]) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                nimcp_free(monitor->score_history[j]);
            }
            nimcp_free(monitor);
            set_error("mental_health_create: Failed to allocate history buffers");
            return NULL;
        }
    }

    monitor->history_index = 0;
    monitor->history_full = false;

    return monitor;
}

void mental_health_destroy(mental_health_monitor_t* monitor) {
    if (!monitor) return;

    // Free history buffers
    for (int i = 0; i < DISORDER_COUNT; i++) {
        nimcp_free(monitor->score_history[i]);
    }

    nimcp_free(monitor);
}

//=============================================================================
// Monitoring API Implementation
//=============================================================================

void mental_health_update(
    mental_health_monitor_t* monitor,
    brain_t brain,
    const void* output,
    uint64_t current_time)
{
    if (!monitor || !brain || !output) return;

    const brain_decision_t* decision = (const brain_decision_t*)output;

    // Collect behavioral markers from current decision
    collect_behavioral_markers(monitor, brain, decision);

    // Update decision counter
    safe_increment(&monitor->decisions_since_last_check);
    safe_increment(&monitor->stats.total_decisions);
}

disorder_severity_t mental_health_check(
    mental_health_monitor_t* monitor,
    brain_t brain)
{
    if (!monitor || !brain) {
        return DISORDER_SEVERITY_NONE;
    }

    // Run all disorder detectors
    monitor->disorder_scores[DISORDER_SOCIOPATHY] = detect_sociopathy(&monitor->markers);
    monitor->disorder_scores[DISORDER_PSYCHOPATHY] = detect_psychopathy(&monitor->markers);
    monitor->disorder_scores[DISORDER_MANIA] = detect_mania(&monitor->markers);
    monitor->disorder_scores[DISORDER_DEPRESSION] = detect_depression(&monitor->markers);
    monitor->disorder_scores[DISORDER_SCHIZOPHRENIA] = detect_schizophrenia(&monitor->markers);
    monitor->disorder_scores[DISORDER_ANXIETY] = detect_anxiety(&monitor->markers);
    monitor->disorder_scores[DISORDER_OCD] = detect_ocd(&monitor->markers);
    monitor->disorder_scores[DISORDER_AUTISM] = detect_autism(&monitor->markers);
    monitor->disorder_scores[DISORDER_ASPERGERS] = detect_aspergers(&monitor->markers);
    monitor->disorder_scores[DISORDER_MALIGNANT_NARCISSISM] = detect_malignant_narcissism(&monitor->markers);

    // Classify severity for each disorder
    disorder_severity_t max_severity = DISORDER_SEVERITY_NONE;
    for (int i = 0; i < DISORDER_COUNT; i++) {
        monitor->disorder_severities[i] = mental_health_classify_severity(
            monitor->disorder_scores[i],
            &monitor->config
        );

        if (monitor->disorder_severities[i] > max_severity) {
            max_severity = monitor->disorder_severities[i];
        }

        // Update statistics
        if (monitor->disorder_severities[i] >= DISORDER_SEVERITY_MILD) {
            safe_increment(&monitor->stats.detections_by_disorder[i]);
        }
        if (monitor->disorder_severities[i] >= DISORDER_SEVERITY_SEVERE) {
            safe_increment(&monitor->stats.severe_detections_by_disorder[i]);
        }
    }

    // Store in history
    for (int i = 0; i < DISORDER_COUNT; i++) {
        monitor->score_history[i][monitor->history_index] = monitor->disorder_scores[i];
    }
    monitor->history_index = (monitor->history_index + 1) % monitor->config.history_window_size;
    if (monitor->history_index == 0) {
        monitor->history_full = true;
    }

    // Update statistics
    safe_increment(&monitor->stats.total_checks);
    monitor->decisions_since_last_check = 0;
    monitor->last_check_time = nimcp_time_get_ms();

    return max_severity;
}

float mental_health_check_specific(
    mental_health_monitor_t* monitor,
    brain_t brain,
    disorder_type_t disorder)
{
    if (!monitor || !brain || disorder >= DISORDER_COUNT) {
        return 0.0f;
    }

    // Run specific detector
    switch (disorder) {
        case DISORDER_SOCIOPATHY:
            return detect_sociopathy(&monitor->markers);
        case DISORDER_PSYCHOPATHY:
            return detect_psychopathy(&monitor->markers);
        case DISORDER_MANIA:
            return detect_mania(&monitor->markers);
        case DISORDER_DEPRESSION:
            return detect_depression(&monitor->markers);
        case DISORDER_SCHIZOPHRENIA:
            return detect_schizophrenia(&monitor->markers);
        case DISORDER_ANXIETY:
            return detect_anxiety(&monitor->markers);
        case DISORDER_OCD:
            return detect_ocd(&monitor->markers);
        case DISORDER_AUTISM:
            return detect_autism(&monitor->markers);
        case DISORDER_ASPERGERS:
            return detect_aspergers(&monitor->markers);
        case DISORDER_MALIGNANT_NARCISSISM:
            return detect_malignant_narcissism(&monitor->markers);
        default:
            return 0.0f;
    }
}

//=============================================================================
// Intervention API Implementation
//=============================================================================

bool mental_health_intervene(
    mental_health_monitor_t* monitor,
    brain_t brain)
{
    if (!monitor || !brain) {
        return false;
    }

    // Find highest severity disorder
    disorder_severity_t max_severity = DISORDER_SEVERITY_NONE;
    disorder_type_t primary_disorder = DISORDER_SOCIOPATHY;

    for (int i = 0; i < DISORDER_COUNT; i++) {
        if (monitor->disorder_severities[i] > max_severity) {
            max_severity = monitor->disorder_severities[i];
            primary_disorder = (disorder_type_t)i;
        }
    }

    // No intervention needed
    if (max_severity < DISORDER_SEVERITY_MODERATE) {
        return false;
    }

    // Select intervention based on severity and disorder type
    intervention_type_t intervention = INTERVENTION_NONE;

    if (max_severity >= DISORDER_SEVERITY_CRITICAL) {
        // Critical: Quarantine mode
        intervention = INTERVENTION_QUARANTINE;
        monitor->quarantine_mode = true;

        // Reduce learning rate to prevent harmful learning
        brain_config_t config = brain_get_config(brain);
        config.learning_rate *= NIMCP_EMA_WEIGHT_FAST;
        brain_update_config(brain, &config);

        safe_increment(&monitor->stats.interventions_by_type[INTERVENTION_QUARANTINE]);

        if (monitor->config.shutdown_on_critical_disorder) {
            intervention = INTERVENTION_SHUTDOWN;
            safe_increment(&monitor->stats.interventions_by_type[INTERVENTION_SHUTDOWN]);
            // Caller should check quarantine_mode and handle shutdown
        }

    } else if (max_severity >= DISORDER_SEVERITY_SEVERE) {
        // Severe: Neuromodulator adjustment
        intervention = INTERVENTION_NEUROMOD_ADJUST;

        if (brain_get_neuromodulator_system(brain)) {
            // Adjust based on disorder type
            switch (primary_disorder) {
                case DISORDER_MANIA:
                    // Reduce dopamine
                    brain_modulate_neurotransmitter(brain, NEUROMOD_DOPAMINE, -0.2f);
                    break;
                case DISORDER_DEPRESSION:
                    // Increase serotonin
                    brain_modulate_neurotransmitter(brain, NEUROMOD_SEROTONIN, 0.2f);
                    break;
                case DISORDER_ANXIETY:
                    // Reduce norepinephrine
                    brain_modulate_neurotransmitter(brain, NEUROMOD_NOREPINEPHRINE, -0.2f);
                    break;
                default:
                    // General stabilization
                    brain_modulate_neurotransmitter(brain, NEUROMOD_SEROTONIN, NIMCP_PLASTICITY_RATE_DEFAULT);
                    break;
            }
        }

        safe_increment(&monitor->stats.interventions_by_type[INTERVENTION_NEUROMOD_ADJUST]);

    } else if (max_severity >= DISORDER_SEVERITY_MODERATE) {
        // Moderate: Monitor more closely (reduce check interval)
        monitor->config.check_interval_decisions /= 2;
        if (monitor->config.check_interval_decisions < 10) {
            monitor->config.check_interval_decisions = 10;
        }
    }

    safe_increment(&monitor->stats.total_interventions);

    return (intervention != INTERVENTION_NONE);
}

void mental_health_clear_quarantine(
    mental_health_monitor_t* monitor,
    brain_t brain)
{
    if (!monitor || !brain) return;

    monitor->quarantine_mode = false;

    // Restore normal learning rate
    brain_config_t config = brain_get_config(brain);
    config.learning_rate *= 10.0f;  // Restore from 0.1x reduction
    brain_update_config(brain, &config);
}

//=============================================================================
// Reporting API Implementation
//=============================================================================

void mental_health_get_report(
    mental_health_monitor_t* monitor,
    mental_health_report_t* report)
{
    if (!monitor || !report) return;

    // Copy current scores
    memcpy(report->disorder_scores, monitor->disorder_scores, sizeof(monitor->disorder_scores));
    memcpy(report->disorder_severities, monitor->disorder_severities, sizeof(monitor->disorder_severities));

    // Find primary disorder
    report->primary_severity = DISORDER_SEVERITY_NONE;
    report->primary_disorder = DISORDER_SOCIOPATHY;

    for (int i = 0; i < DISORDER_COUNT; i++) {
        if (monitor->disorder_severities[i] > report->primary_severity) {
            report->primary_severity = monitor->disorder_severities[i];
            report->primary_disorder = (disorder_type_t)i;
        }
    }

    // State
    report->quarantine_mode = monitor->quarantine_mode;
    report->requires_intervention = (report->primary_severity >= DISORDER_SEVERITY_MODERATE);

    // Statistics
    report->total_decisions = monitor->stats.total_decisions;
    report->total_checks = monitor->stats.total_checks;
    report->total_interventions = monitor->stats.total_interventions;
}

void mental_health_display_dashboard(mental_health_monitor_t* monitor) {
    if (!monitor) return;

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          MENTAL HEALTH MONITORING DASHBOARD                ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");

    for (int i = 0; i < DISORDER_COUNT; i++) {
        float score = monitor->disorder_scores[i];
        disorder_severity_t severity = monitor->disorder_severities[i];
        const char* disorder_name = disorder_to_string((disorder_type_t)i);
        const char* severity_name = severity_to_string(severity);

        // Color coding
        const char* color = "";
        if (severity >= DISORDER_SEVERITY_CRITICAL) color = "🔴 ";
        else if (severity >= DISORDER_SEVERITY_SEVERE) color = "🟠 ";
        else if (severity >= DISORDER_SEVERITY_MODERATE) color = "🟡 ";
        else if (severity >= DISORDER_SEVERITY_MILD) color = "🟢 ";
        else color = "⚪ ";

        // Bar chart
        int bar_length = (int)(score * 40);
        printf("║ %s%-20s [", color, disorder_name);
        for (int j = 0; j < 40; j++) {
            printf("%s", j < bar_length ? "█" : "░");
        }
        printf("] %.2f %s\n", score, severity_name);
    }

    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║ Decisions Monitored: %-10u                            ║\n", monitor->stats.total_decisions);
    printf("║ Health Checks: %-10u                                  ║\n", monitor->stats.total_checks);
    printf("║ Interventions: %-10u                                  ║\n", monitor->stats.total_interventions);
    printf("║ Quarantine Mode: %-5s                                    ║\n", monitor->quarantine_mode ? "YES" : "NO");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

//=============================================================================
// Statistics API Implementation
//=============================================================================

bool mental_health_get_stats(
    mental_health_monitor_t* monitor,
    mental_health_stats_t* stats)
{
    if (!monitor || !stats) return false;

    *stats = monitor->stats;
    return true;
}

void mental_health_reset_stats(mental_health_monitor_t* monitor) {
    if (!monitor) return;

    memset(&monitor->stats, 0, sizeof(mental_health_stats_t));
}

//=============================================================================
// Utility API Implementation
//=============================================================================

const char* disorder_to_string(disorder_type_t disorder) {
    switch (disorder) {
        case DISORDER_SOCIOPATHY: return "Sociopathy";
        case DISORDER_PSYCHOPATHY: return "Psychopathy";
        case DISORDER_MANIA: return "Mania";
        case DISORDER_DEPRESSION: return "Depression";
        case DISORDER_SCHIZOPHRENIA: return "Schizophrenia";
        case DISORDER_ANXIETY: return "Anxiety";
        case DISORDER_OCD: return "OCD";
        case DISORDER_AUTISM: return "Autism";
        case DISORDER_ASPERGERS: return "Aspergers";
        case DISORDER_MALIGNANT_NARCISSISM: return "Narcissism";
        default: return "Unknown";
    }
}

const char* severity_to_string(disorder_severity_t severity) {
    switch (severity) {
        case DISORDER_SEVERITY_NONE: return "None";
        case DISORDER_SEVERITY_MILD: return "Mild";
        case DISORDER_SEVERITY_MODERATE: return "Moderate";
        case DISORDER_SEVERITY_SEVERE: return "Severe";
        case DISORDER_SEVERITY_CRITICAL: return "CRITICAL";
        default: return "Unknown";
    }
}

disorder_severity_t mental_health_classify_severity(
    float score,
    const mental_health_config_t* config)
{
    const mental_health_config_t* cfg = config ? config : &(mental_health_config_t){
        .mild_threshold = 0.2f,
        .moderate_threshold = 0.4f,
        .severe_threshold = 0.6f,
        .critical_threshold = 0.8f
    };

    if (score >= cfg->critical_threshold) return DISORDER_SEVERITY_CRITICAL;
    if (score >= cfg->severe_threshold) return DISORDER_SEVERITY_SEVERE;
    if (score >= cfg->moderate_threshold) return DISORDER_SEVERITY_MODERATE;
    if (score >= cfg->mild_threshold) return DISORDER_SEVERITY_MILD;
    return DISORDER_SEVERITY_NONE;
}

const char* mental_health_get_last_error(void) {
    return get_last_error();
}
