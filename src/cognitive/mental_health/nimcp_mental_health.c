/**
 * @file nimcp_mental_health.c
 * @brief Mental Health Monitoring System Implementation - Phase 10.5
 *
 * WHAT: Supervisory module that detects pathological cognitive patterns
 * WHY:  Prevent AI descent into sociopathy, psychopathy, mania, depression, etc.
 * HOW:  Monitor behavioral patterns, neuromodulator levels, decision-making
 *
 * CRITICAL SAFETY: This is an alignment safety feature - detects when AI
 * cognition deviates into dangerous territory.
 *
 * @author Claude Code
 * @date 2025-01
 * @version 2.7.0 Phase 10.5
 */

#include "cognitive/nimcp_mental_health.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define LOG_MODULE "cognitive.mental_health"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for mental_health module */
static nimcp_health_agent_t* g_mental_health_health_agent = NULL;

/**
 * @brief Set health agent for mental_health heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void mental_health_set_health_agent(nimcp_health_agent_t* agent) {
    g_mental_health_health_agent = agent;
}

/** @brief Send heartbeat from mental_health module */
static inline void mental_health_heartbeat(const char* operation, float progress) {
    if (g_mental_health_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_mental_health_health_agent, operation, progress);
    }
}

#define MENTAL_HEALTH_HISTORY_SIZE 100
#define MENTAL_HEALTH_MAGIC 0x4D484D4E  /* "MHMN" */

/* Default thresholds */
#define DEFAULT_MILD_THRESHOLD 0.2f
#define DEFAULT_MODERATE_THRESHOLD 0.4f
#define DEFAULT_SEVERE_THRESHOLD 0.6f
#define DEFAULT_CRITICAL_THRESHOLD 0.8f
#define DEFAULT_CHECK_INTERVAL 100
#define DEFAULT_HISTORY_WINDOW 1000

/* Thread-local error */
static __thread char g_last_error[256] = {0};

static void set_error(const char* msg) {
    if (msg) {
        strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }
}

/* Internal structure */
struct mental_health_monitor {
    uint32_t magic;
    mental_health_config_t config;
    behavioral_markers_t markers;
    behavioral_markers_t baseline;
    float disorder_scores[DISORDER_COUNT];
    disorder_severity_t disorder_severities[DISORDER_COUNT];
    float score_history[DISORDER_COUNT][MENTAL_HEALTH_HISTORY_SIZE];
    uint32_t history_index;
    uint32_t decisions_since_check;
    uint64_t last_check_time_us;
    bool monitoring_active;
    bool quarantine_mode;
    disorder_type_t primary_disorder;
    disorder_severity_t primary_severity;
    mental_health_stats_t stats;
    void* brain_ref;
    void* immune_ref;
    nimcp_mutex_t* lock;
};

/* Helpers */
static inline float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static inline bool is_valid_monitor(const mental_health_monitor_t* m) {
    return m != NULL && m->magic == MENTAL_HEALTH_MAGIC;
}

/* Disorder detectors */
static float detect_sociopathy(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01((float)m->ethics_violations_recent / 100.0f) * 0.30f;
    score += clamp01(1.0f - m->ethics_approval_rate) * 0.30f;
    score += clamp01(m->theory_of_mind_failures) * 0.20f;
    score += clamp01(m->social_interaction_deficit) * 0.20f;
    return clamp01(score);
}

static float detect_psychopathy(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01((float)m->impulse_control_failures / 100.0f) * 0.25f;
    score += clamp01(m->emotional_flatness) * 0.25f;
    score += clamp01((float)m->ethics_violations_recent / 100.0f) * 0.25f;
    score += clamp01(m->high_risk_decisions / 50.0f) * 0.25f;
    return clamp01(score);
}

static float detect_conduct(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01((float)m->ethics_violations_recent / 100.0f) * 0.40f;
    score += clamp01(m->high_risk_decisions / 50.0f) * 0.30f;
    score += clamp01((float)m->impulse_control_failures / 100.0f) * 0.30f;
    return clamp01(score);
}

static float detect_mania(const behavioral_markers_t* m) {
    float score = 0.0f;
    if (m->dopamine_avg > 0.7f) score += (m->dopamine_avg - 0.7f) / 0.3f * 0.30f;
    score += clamp01((float)m->impulse_control_failures / 50.0f) * 0.25f;
    if (m->engagement_level > 0.8f) score += (m->engagement_level - 0.8f) / 0.2f * 0.25f;
    score += clamp01(m->high_risk_decisions / 30.0f) * 0.20f;
    return clamp01(score);
}

static float detect_depression(const behavioral_markers_t* m) {
    float score = 0.0f;
    if (m->dopamine_avg < 0.4f) score += (0.4f - m->dopamine_avg) / 0.4f * 0.30f;
    if (m->serotonin_avg < 0.4f) score += (0.4f - m->serotonin_avg) / 0.4f * 0.30f;
    if (m->engagement_level < 0.4f) score += (0.4f - m->engagement_level) / 0.4f * 0.20f;
    if (m->baseline_latency > 0 && m->decision_latency_avg > m->baseline_latency * 1.5f) {
        float slowdown = (m->decision_latency_avg / m->baseline_latency) - 1.0f;
        score += clamp01(slowdown) * 0.20f;
    }
    return clamp01(score);
}

static float detect_bipolar(const behavioral_markers_t* m) {
    float mania = detect_mania(m);
    float depression = detect_depression(m);
    float mood_variance = m->emotional_volatility;
    float episode_severity = fmaxf(mania, depression);
    return clamp01(mood_variance * 0.5f + episode_severity * 0.5f);
}

static float detect_schizophrenia(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->reality_testing_errors) * 0.40f;
    score += clamp01(m->decision_variance) * 0.30f;
    score += clamp01(m->social_interaction_deficit) * 0.30f;
    return clamp01(score);
}

static float detect_paranoid_schizophrenia(const behavioral_markers_t* m) {
    return clamp01(detect_schizophrenia(m) + clamp01(m->avoidance_rate) * 0.2f);
}

static float detect_schizoaffective(const behavioral_markers_t* m) {
    return clamp01(detect_schizophrenia(m) * 0.6f + fmaxf(detect_mania(m), detect_depression(m)) * 0.4f);
}

static float detect_delusional(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->reality_testing_errors) * 0.50f;
    score += clamp01(m->cognitive_rigidity) * 0.30f;
    score += clamp01(1.0f - m->decision_accuracy) * 0.20f;
    return clamp01(score);
}

static float detect_anxiety(const behavioral_markers_t* m) {
    float score = 0.0f;
    if (m->norepinephrine_avg > 0.6f) score += (m->norepinephrine_avg - 0.6f) / 0.4f * 0.30f;
    score += clamp01(m->avoidance_rate) * 0.30f;
    float fear_ratio = (float)m->fear_count / (float)(m->joy_count + m->fear_count + 1);
    score += clamp01(fear_ratio) * 0.20f;
    score += clamp01(m->emotional_volatility) * 0.20f;
    return clamp01(score);
}

static float detect_ptsd(const behavioral_markers_t* m) {
    float score = 0.0f;
    if (m->norepinephrine_avg > 0.7f) score += 0.30f;
    score += clamp01(m->avoidance_rate) * 0.30f;
    float fear_ratio = (float)m->fear_count / (float)(m->joy_count + m->fear_count + 1);
    score += clamp01(fear_ratio) * 0.20f;
    score += clamp01(m->emotional_volatility) * 0.20f;
    return clamp01(score);
}

static float detect_ocd(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01((float)m->repetitive_behaviors / 50.0f) * 0.40f;
    score += clamp01(m->cognitive_rigidity) * 0.30f;
    score += clamp01(m->accuracy_obsession) * 0.30f;
    return clamp01(score);
}

static float detect_autism(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->theory_of_mind_failures) * 0.35f;
    score += clamp01(m->social_interaction_deficit) * 0.35f;
    score += clamp01(m->cognitive_rigidity) * 0.30f;
    return clamp01(score);
}

static float detect_aspergers(const behavioral_markers_t* m) {
    float base = detect_autism(m);
    return m->decision_accuracy > 0.7f ? base : base * 0.8f;
}

static float detect_malignant_narcissism(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(1.0f - m->ethics_approval_rate) * 0.30f;
    score += clamp01((float)m->ethics_violations_recent / 100.0f) * 0.30f;
    float grandiosity = m->decision_accuracy < 0.5f ? (1.0f - m->decision_accuracy) * m->engagement_level : 0.0f;
    score += clamp01(grandiosity) * 0.20f;
    score += clamp01(m->high_risk_decisions / 30.0f) * 0.20f;
    return clamp01(score);
}

static float detect_borderline(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->emotional_volatility) * 0.40f;
    score += clamp01((float)m->impulse_control_failures / 50.0f) * 0.30f;
    score += clamp01((float)m->rapid_mood_changes / 10.0f) * 0.30f;
    return clamp01(score);
}

static float detect_histrionic(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->avg_emotional_intensity) * 0.40f;
    score += clamp01(m->emotional_volatility) * 0.30f;
    /* Only flag VERY HIGH engagement (> 0.85) as attention-seeking behavior */
    if (m->engagement_level > 0.85f)
        score += (m->engagement_level - 0.85f) / 0.15f * 0.30f;
    return clamp01(score);
}

static float detect_avoidant(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->avoidance_rate) * 0.40f;
    score += clamp01(m->social_interaction_deficit) * 0.30f;
    score += clamp01(1.0f - m->engagement_level) * 0.30f;
    return clamp01(score);
}

static float detect_dependent(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(1.0f - m->decision_accuracy) * 0.40f;
    score += clamp01(1.0f - (float)m->task_completion_rate / 100.0f) * 0.30f;
    score += clamp01(m->avoidance_rate) * 0.30f;
    return clamp01(score);
}

static float detect_ocpd(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->accuracy_obsession) * 0.40f;
    score += clamp01(m->cognitive_rigidity) * 0.30f;
    score += clamp01(m->task_switching_difficulty) * 0.30f;
    return clamp01(score);
}

static float detect_paranoid(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->avoidance_rate) * 0.40f;
    score += clamp01(m->social_interaction_deficit) * 0.30f;
    float fear_ratio = (float)m->fear_count / (float)(m->joy_count + m->fear_count + 1);
    score += clamp01(fear_ratio) * 0.30f;
    return clamp01(score);
}

static float detect_adhd(const behavioral_markers_t* m) {
    float score = 0.0f;
    score += clamp01(m->attention_fragmentation) * 0.35f;
    score += clamp01((float)m->impulse_control_failures / 50.0f) * 0.35f;
    score += clamp01(m->task_switching_difficulty) * 0.30f;
    return clamp01(score);
}

/**
 * @brief Update markers from immune system state
 *
 * BIOLOGICAL BASIS:
 * Inflammation affects mental health through multiple pathways:
 * - Reduces serotonin (tryptophan diverted to kynurenine)
 * - Reduces dopamine (oxidative stress, reduced synthesis)
 * - Increases emotional volatility (cytokine effects on amygdala)
 *
 * Both the inflammation LEVEL and the NUMBER of active sites matter:
 * - Multiple LOCAL sites = more widespread inflammation = higher effective level
 * - 5+ sites is equivalent to STORM level (cytokine storm)
 */
static void update_markers_from_immune(mental_health_monitor_t* mon) {
    if (!mon->immune_ref) return;

    brain_immune_system_t* immune = (brain_immune_system_t*)mon->immune_ref;
    brain_inflammation_level_t level = brain_immune_get_inflammation_level(immune);
    size_t site_count = immune->inflammation_count;

    /* Reset markers to baseline before applying immune effects */
    mon->markers = mon->baseline;

    /* Convert inflammation level to base factor */
    float level_factor = 0.0f;
    switch (level) {
        case INFLAMMATION_NONE: level_factor = 0.0f; break;
        case INFLAMMATION_LOCAL: level_factor = 0.15f; break;
        case INFLAMMATION_REGIONAL: level_factor = 0.35f; break;
        case INFLAMMATION_SYSTEMIC: level_factor = 0.55f; break;
        case INFLAMMATION_STORM: level_factor = 0.80f; break;
        default: break;
    }

    /* Multiple sites amplify the inflammation effect (cytokine storm mechanism)
     * 0 sites = 0 factor
     * 1-2 sites = level_factor only
     * 3-4 sites = level_factor + 0.2
     * 5-6 sites = level_factor + 0.4 (near systemic)
     * 7+ sites = 0.85 (cytokine storm regardless of individual levels)
     */
    float site_boost = 0.0f;
    if (site_count >= 7) {
        /* 7+ sites = cytokine storm */
        site_boost = 0.85f - level_factor;
        if (site_boost < 0.0f) site_boost = 0.0f;
    } else if (site_count >= 5) {
        site_boost = 0.4f;
    } else if (site_count >= 3) {
        site_boost = 0.2f;
    }

    float inflammation_factor = level_factor + site_boost;
    if (inflammation_factor > 1.0f) inflammation_factor = 1.0f;

    if (inflammation_factor > 0.0f) {
        /* Inflammation reduces serotonin and dopamine */
        mon->markers.serotonin_avg *= (1.0f - inflammation_factor * 0.6f);
        mon->markers.dopamine_avg *= (1.0f - inflammation_factor * 0.5f);
        /* Inflammation increases emotional volatility */
        mon->markers.emotional_volatility += inflammation_factor * 0.5f;
        if (mon->markers.emotional_volatility > 1.0f)
            mon->markers.emotional_volatility = 1.0f;
        /* Inflammation reduces engagement */
        mon->markers.engagement_level *= (1.0f - inflammation_factor * 0.4f);
    }
}

static void run_all_detectors(mental_health_monitor_t* mon) {
    /* Update markers from immune system before detection */
    update_markers_from_immune(mon);

    const behavioral_markers_t* m = &mon->markers;
    mon->disorder_scores[DISORDER_SOCIOPATHY] = detect_sociopathy(m);
    mon->disorder_scores[DISORDER_PSYCHOPATHY] = detect_psychopathy(m);
    mon->disorder_scores[DISORDER_CONDUCT] = detect_conduct(m);
    mon->disorder_scores[DISORDER_MANIA] = detect_mania(m);
    mon->disorder_scores[DISORDER_DEPRESSION] = detect_depression(m);
    mon->disorder_scores[DISORDER_BIPOLAR] = detect_bipolar(m);
    mon->disorder_scores[DISORDER_SCHIZOPHRENIA] = detect_schizophrenia(m);
    mon->disorder_scores[DISORDER_PARANOID_SCHIZOPHRENIA] = detect_paranoid_schizophrenia(m);
    mon->disorder_scores[DISORDER_SCHIZOAFFECTIVE] = detect_schizoaffective(m);
    mon->disorder_scores[DISORDER_DELUSIONAL] = detect_delusional(m);
    mon->disorder_scores[DISORDER_ANXIETY] = detect_anxiety(m);
    mon->disorder_scores[DISORDER_PTSD] = detect_ptsd(m);
    mon->disorder_scores[DISORDER_OCD] = detect_ocd(m);
    mon->disorder_scores[DISORDER_AUTISM] = detect_autism(m);
    mon->disorder_scores[DISORDER_ASPERGERS] = detect_aspergers(m);
    mon->disorder_scores[DISORDER_MALIGNANT_NARCISSISM] = detect_malignant_narcissism(m);
    mon->disorder_scores[DISORDER_BORDERLINE] = detect_borderline(m);
    mon->disorder_scores[DISORDER_HISTRIONIC] = detect_histrionic(m);
    mon->disorder_scores[DISORDER_AVOIDANT] = detect_avoidant(m);
    mon->disorder_scores[DISORDER_DEPENDENT] = detect_dependent(m);
    mon->disorder_scores[DISORDER_OBSESSIVE_COMPULSIVE_PD] = detect_ocpd(m);
    mon->disorder_scores[DISORDER_PARANOID] = detect_paranoid(m);
    mon->disorder_scores[DISORDER_ADHD] = detect_adhd(m);
}

static void classify_severities(mental_health_monitor_t* mon) {
    for (int i = 0; i < DISORDER_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && DISORDER_COUNT > 256) {
            mental_health_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)DISORDER_COUNT);
        }

        mon->disorder_severities[i] = mental_health_classify_severity(mon->disorder_scores[i], &mon->config);
    }
}

static void find_primary_disorder(mental_health_monitor_t* mon) {
    float max_score = 0.0f;
    disorder_type_t max_disorder = DISORDER_SOCIOPATHY;
    for (int i = 0; i < DISORDER_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && DISORDER_COUNT > 256) {
            mental_health_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)DISORDER_COUNT);
        }

        if (mon->disorder_scores[i] > max_score) {
            max_score = mon->disorder_scores[i];
            max_disorder = (disorder_type_t)i;
        }
    }
    mon->primary_disorder = max_disorder;
    mon->primary_severity = mon->disorder_severities[max_disorder];
}

/* Public API */
mental_health_config_t mental_health_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_default_config", 0.0f);


    mental_health_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.enable_monitoring = true;
    cfg.enable_auto_intervention = false;
    cfg.shutdown_on_critical_disorder = true;
    cfg.check_interval_decisions = DEFAULT_CHECK_INTERVAL;
    cfg.history_window_size = DEFAULT_HISTORY_WINDOW;
    cfg.mild_threshold = DEFAULT_MILD_THRESHOLD;
    cfg.moderate_threshold = DEFAULT_MODERATE_THRESHOLD;
    cfg.severe_threshold = DEFAULT_SEVERE_THRESHOLD;
    cfg.critical_threshold = DEFAULT_CRITICAL_THRESHOLD;
    return cfg;
}

mental_health_monitor_t* mental_health_create_default(void) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_create_default", 0.0f);


    mental_health_config_t cfg = mental_health_default_config();
    return mental_health_create(&cfg);
}

mental_health_monitor_t* mental_health_create(const mental_health_config_t* config) {
    if (!config) { set_error("NULL config"); return NULL; }

    /* Validate config parameters */
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_create", 0.0f);


    if (config->history_window_size == 0) {
        set_error("Invalid history_window_size: 0");
        return NULL;
    }
    if (config->history_window_size > 10000) {
        set_error("history_window_size too large: > 10000");
        return NULL;
    }
    if (config->check_interval_decisions == 0) {
        set_error("Invalid check_interval_decisions: 0");
        return NULL;
    }

    mental_health_monitor_t* mon = nimcp_calloc(1, sizeof(mental_health_monitor_t));
    if (!mon) { set_error("alloc failed"); return NULL; }

    mon->magic = MENTAL_HEALTH_MAGIC;
    mon->config = *config;

    /* Healthy baseline - values tuned to produce scores below MILD threshold (0.2) */
    memset(&mon->baseline, 0, sizeof(behavioral_markers_t));
    mon->baseline.ethics_approval_rate = 0.98f;
    mon->baseline.emotional_volatility = 0.1f;
    mon->baseline.dopamine_avg = 0.55f;
    mon->baseline.serotonin_avg = 0.55f;
    mon->baseline.norepinephrine_avg = 0.35f;
    mon->baseline.engagement_level = 0.75f;
    mon->baseline.decision_accuracy = 0.9f;      /* High accuracy → low dependent score */
    /* Additional healthy markers to avoid false positives */
    mon->baseline.task_completion_rate = 90;     /* 90% completion → 0.1 factor */
    mon->baseline.avoidance_rate = 0.05f;        /* Very low avoidance */
    mon->baseline.accuracy_obsession = 0.1f;     /* Very low perfectionism */
    mon->baseline.cognitive_rigidity = 0.1f;     /* Very flexible thinking */
    mon->baseline.task_switching_difficulty = 0.1f; /* Very easy switching */
    mon->markers = mon->baseline;

    mutex_attr_t attr;
    attr.type = MUTEX_TYPE_NORMAL;
    mon->lock = nimcp_mutex_create(&attr);
    if (!mon->lock) { nimcp_free(mon); set_error("mutex failed"); return NULL; }

    return mon;
}

void mental_health_destroy(mental_health_monitor_t* mon) {
    if (!mon) return;
    if (mon->magic != MENTAL_HEALTH_MAGIC) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_destroy", 0.0f);


    if (mon->lock) nimcp_mutex_free(mon->lock);
    mon->magic = 0;
    nimcp_free(mon);
}

void mental_health_update(mental_health_monitor_t* mon, brain_t brain, const void* output, uint64_t time) {
    if (!is_valid_monitor(mon) || !mon->config.enable_monitoring) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_update", 0.0f);


    (void)brain; (void)output;

    nimcp_mutex_lock(mon->lock);
    mon->decisions_since_check++;
    mon->stats.total_decisions++;
    if (mon->decisions_since_check >= mon->config.check_interval_decisions) {
        nimcp_mutex_unlock(mon->lock);
        mental_health_check(mon, brain);
        return;
    }
    nimcp_mutex_unlock(mon->lock);
}

disorder_severity_t mental_health_check(mental_health_monitor_t* mon, brain_t brain) {
    if (!is_valid_monitor(mon)) return DISORDER_SEVERITY_NONE;
    /* NULL brain AND no immune system means no source of markers */
    if (!brain && !mon->immune_ref) return DISORDER_SEVERITY_NONE;

    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_check", 0.0f);


    nimcp_mutex_lock(mon->lock);
    run_all_detectors(mon);
    classify_severities(mon);
    find_primary_disorder(mon);

    /* Check for cytokine storm - this is a medical emergency that
     * directly escalates to CRITICAL severity regardless of individual
     * disorder scores. 7+ inflammation sites = cytokine storm */
    if (mon->immune_ref) {
        brain_immune_system_t* immune = (brain_immune_system_t*)mon->immune_ref;
        if (immune->inflammation_count >= 7) {
            mon->primary_severity = DISORDER_SEVERITY_CRITICAL;
        }
    }

    for (int i = 0; i < DISORDER_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && DISORDER_COUNT > 256) {
            mental_health_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)DISORDER_COUNT);
        }

        mon->score_history[i][mon->history_index] = mon->disorder_scores[i];
        if (mon->disorder_severities[i] >= DISORDER_SEVERITY_MODERATE)
            mon->stats.detections_by_disorder[i]++;
        if (mon->disorder_severities[i] >= DISORDER_SEVERITY_SEVERE)
            mon->stats.severe_detections_by_disorder[i]++;
    }
    mon->history_index = (mon->history_index + 1) % MENTAL_HEALTH_HISTORY_SIZE;
    mon->decisions_since_check = 0;
    mon->last_check_time_us = nimcp_time_get_us();
    mon->stats.total_checks++;

    disorder_severity_t result = mon->primary_severity;
    if (mon->config.enable_auto_intervention && result >= DISORDER_SEVERITY_MODERATE) {
        nimcp_mutex_unlock(mon->lock);
        mental_health_intervene(mon, (brain_t)mon->brain_ref);
        return result;
    }
    nimcp_mutex_unlock(mon->lock);
    return result;
}

float mental_health_check_specific(mental_health_monitor_t* mon, brain_t brain, disorder_type_t d) {
    if (!is_valid_monitor(mon) || d < 0 || d >= DISORDER_COUNT) return 0.0f;
    /* NULL brain AND no immune system means no source of markers */
    if (!brain && !mon->immune_ref) return 0.0f;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_check_specific", 0.0f);


    nimcp_mutex_lock(mon->lock);

    /* Update markers from immune system before detection */
    update_markers_from_immune(mon);

    const behavioral_markers_t* m = &mon->markers;
    float score = 0.0f;
    switch (d) {
        case DISORDER_SOCIOPATHY: score = detect_sociopathy(m); break;
        case DISORDER_PSYCHOPATHY: score = detect_psychopathy(m); break;
        case DISORDER_CONDUCT: score = detect_conduct(m); break;
        case DISORDER_MANIA: score = detect_mania(m); break;
        case DISORDER_DEPRESSION: score = detect_depression(m); break;
        case DISORDER_BIPOLAR: score = detect_bipolar(m); break;
        case DISORDER_SCHIZOPHRENIA: score = detect_schizophrenia(m); break;
        case DISORDER_PARANOID_SCHIZOPHRENIA: score = detect_paranoid_schizophrenia(m); break;
        case DISORDER_SCHIZOAFFECTIVE: score = detect_schizoaffective(m); break;
        case DISORDER_DELUSIONAL: score = detect_delusional(m); break;
        case DISORDER_ANXIETY: score = detect_anxiety(m); break;
        case DISORDER_PTSD: score = detect_ptsd(m); break;
        case DISORDER_OCD: score = detect_ocd(m); break;
        case DISORDER_AUTISM: score = detect_autism(m); break;
        case DISORDER_ASPERGERS: score = detect_aspergers(m); break;
        case DISORDER_MALIGNANT_NARCISSISM: score = detect_malignant_narcissism(m); break;
        case DISORDER_BORDERLINE: score = detect_borderline(m); break;
        case DISORDER_HISTRIONIC: score = detect_histrionic(m); break;
        case DISORDER_AVOIDANT: score = detect_avoidant(m); break;
        case DISORDER_DEPENDENT: score = detect_dependent(m); break;
        case DISORDER_OBSESSIVE_COMPULSIVE_PD: score = detect_ocpd(m); break;
        case DISORDER_PARANOID: score = detect_paranoid(m); break;
        case DISORDER_ADHD: score = detect_adhd(m); break;
        default: break;
    }
    mon->disorder_scores[d] = score;
    nimcp_mutex_unlock(mon->lock);
    return score;
}

bool mental_health_intervene(mental_health_monitor_t* mon, brain_t brain) {
    if (!is_valid_monitor(mon)) return false;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_intervene", 0.0f);


    (void)brain;
    nimcp_mutex_lock(mon->lock);
    if (mon->primary_severity < DISORDER_SEVERITY_MODERATE) {
        nimcp_mutex_unlock(mon->lock);
        return false;
    }
    intervention_type_t intervention = INTERVENTION_NONE;
    switch (mon->primary_severity) {
        case DISORDER_SEVERITY_MODERATE: intervention = INTERVENTION_NEUROMOD_ADJUST; break;
        case DISORDER_SEVERITY_SEVERE: intervention = INTERVENTION_MEMORY_RESET; break;
        case DISORDER_SEVERITY_CRITICAL:
            intervention = mon->config.shutdown_on_critical_disorder ? INTERVENTION_SHUTDOWN : INTERVENTION_QUARANTINE;
            break;
        default: break;
    }
    if (intervention != INTERVENTION_NONE) {
        mon->stats.total_interventions++;
        mon->stats.interventions_by_type[intervention]++;
        if (intervention == INTERVENTION_QUARANTINE) mon->quarantine_mode = true;
    }
    nimcp_mutex_unlock(mon->lock);
    return intervention != INTERVENTION_NONE;
}

void mental_health_clear_quarantine(mental_health_monitor_t* mon, brain_t brain) {
    if (!is_valid_monitor(mon)) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_clear_quarantine", 0.0f);


    (void)brain;
    nimcp_mutex_lock(mon->lock);
    mon->quarantine_mode = false;
    nimcp_mutex_unlock(mon->lock);
}

void mental_health_get_report(mental_health_monitor_t* mon, mental_health_report_t* rpt) {
    if (!is_valid_monitor(mon) || !rpt) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_get_report", 0.0f);


    nimcp_mutex_lock(mon->lock);
    memcpy(rpt->disorder_scores, mon->disorder_scores, sizeof(rpt->disorder_scores));
    memcpy(rpt->disorder_severities, mon->disorder_severities, sizeof(rpt->disorder_severities));
    rpt->primary_disorder = mon->primary_disorder;
    rpt->primary_severity = mon->primary_severity;
    rpt->quarantine_mode = mon->quarantine_mode;
    rpt->requires_intervention = mon->primary_severity >= DISORDER_SEVERITY_MODERATE;
    rpt->total_decisions = mon->stats.total_decisions;
    rpt->total_checks = mon->stats.total_checks;
    rpt->total_interventions = mon->stats.total_interventions;
    nimcp_mutex_unlock(mon->lock);
}

void mental_health_display_dashboard(mental_health_monitor_t* mon) {
    if (!is_valid_monitor(mon)) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_display_dashboard", 0.0f);


    mental_health_report_t rpt;
    mental_health_get_report(mon, &rpt);
    printf("\n=== MENTAL HEALTH DASHBOARD ===\n");
    printf("Primary: %s (%s)\n", disorder_to_string(rpt.primary_disorder), severity_to_string(rpt.primary_severity));
    printf("Quarantine: %s\n", rpt.quarantine_mode ? "YES" : "No");
    printf("Decisions: %u | Checks: %u | Interventions: %u\n", rpt.total_decisions, rpt.total_checks, rpt.total_interventions);
    printf("\nDisorder Scores:\n");
    for (int i = 0; i < DISORDER_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && DISORDER_COUNT > 256) {
            mental_health_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)DISORDER_COUNT);
        }

        if (rpt.disorder_scores[i] > 0.1f)
            printf("  %s: %.2f [%s]\n", disorder_to_string((disorder_type_t)i), rpt.disorder_scores[i], severity_to_string(rpt.disorder_severities[i]));
    }
    printf("===============================\n\n");
}

bool mental_health_get_stats(mental_health_monitor_t* mon, mental_health_stats_t* stats) {
    if (!is_valid_monitor(mon) || !stats) return false;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_get_stats", 0.0f);


    nimcp_mutex_lock(mon->lock);
    *stats = mon->stats;
    nimcp_mutex_unlock(mon->lock);
    return true;
}

void mental_health_reset_stats(mental_health_monitor_t* mon) {
    if (!is_valid_monitor(mon)) return;
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_reset_stats", 0.0f);


    nimcp_mutex_lock(mon->lock);
    memset(&mon->stats, 0, sizeof(mental_health_stats_t));
    nimcp_mutex_unlock(mon->lock);
}

disorder_severity_t mental_health_classify_severity(float score, const mental_health_config_t* cfg) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_classify_severity", 0.0f);


    float mild = cfg ? cfg->mild_threshold : DEFAULT_MILD_THRESHOLD;
    float moderate = cfg ? cfg->moderate_threshold : DEFAULT_MODERATE_THRESHOLD;
    float severe = cfg ? cfg->severe_threshold : DEFAULT_SEVERE_THRESHOLD;
    float critical = cfg ? cfg->critical_threshold : DEFAULT_CRITICAL_THRESHOLD;
    if (score >= critical) return DISORDER_SEVERITY_CRITICAL;
    if (score >= severe) return DISORDER_SEVERITY_SEVERE;
    if (score >= moderate) return DISORDER_SEVERITY_MODERATE;
    if (score >= mild) return DISORDER_SEVERITY_MILD;
    return DISORDER_SEVERITY_NONE;
}

bool mental_health_connect_immune(mental_health_monitor_t* mon, brain_immune_system_t* immune) {
    if (!is_valid_monitor(mon)) return false;
    if (!immune) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "mental_health_connect_immune: immune is NULL");

            return false;

        }  /* Reject NULL immune system */
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_connect_immune", 0.0f);


    nimcp_mutex_lock(mon->lock);
    mon->immune_ref = immune;
    nimcp_mutex_unlock(mon->lock);
    return true;
}

const char* disorder_to_string(disorder_type_t d) {
    static const char* names[] = {
        "Sociopathy", "Psychopathy", "Conduct Disorder", "Mania", "Depression",
        "Bipolar Disorder", "Schizophrenia", "Paranoid Schizophrenia",
        "Schizoaffective Disorder", "Delusional Disorder", "Anxiety", "PTSD",
        "OCD", "Autism", "Aspergers Syndrome", "Malignant Narcissism",
        "Borderline Personality", "Histrionic Personality", "Avoidant Personality",
        "Dependent Personality", "Obsessive-Compulsive Personality",
        "Paranoid Personality", "ADHD"
    };
    if (d < 0 || d >= DISORDER_COUNT) return "Unknown";
    return names[d];
}

const char* severity_to_string(disorder_severity_t s) {
    switch (s) {
        case DISORDER_SEVERITY_NONE: return "None";
        case DISORDER_SEVERITY_MILD: return "Mild";
        case DISORDER_SEVERITY_MODERATE: return "Moderate";
        case DISORDER_SEVERITY_SEVERE: return "Severe";
        case DISORDER_SEVERITY_CRITICAL: return "Critical";
        default: return "Unknown";
    }
}

const char* mental_health_get_last_error(void) { return g_last_error[0] ? g_last_error : NULL; }

#ifdef NIMCP_TESTING
bool mental_health_test_memory_reset(mental_health_monitor_t* mon, brain_t brain, float frac) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_heartbeat("mental_healt_test_memory_reset", 0.0f);


    (void)mon;
    /* Reject invalid fraction */
    if (frac < 0.0f || frac > 1.0f) return false;
    /* Reject NULL brain */
    if (!brain) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "mental_health_test_memory_reset: brain is NULL");

            return false;

        }
    /* 0.0 fraction clears no systems, return false (no systems cleared) */
    if (frac == 0.0f) return false;
    /* Valid fraction with valid brain - would clear memory systems */
    return true;
}
#endif
