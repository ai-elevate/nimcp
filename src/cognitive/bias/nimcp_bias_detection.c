/* SPDX-License-Identifier: MIT */
/**
 * @file nimcp_bias_detection.c
 * @brief Phase E6: Bias Detection and Correction Implementation
 *
 * Implements comprehensive bias detection enabling NIMCP to:
 * 1. Self-monitor implicit and explicit biases
 * 2. Detect biases in humans during interactions
 * 3. Apply evidence-based debiasing interventions
 * 4. Ensure statistical fairness in decision-making
 *
 * @author Claude Code (Anthropic)
 * @date 2025-11-13
 */

#include "cognitive/nimcp_bias_detection.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline float exponential_decay(float value, float rate, float dt) {
    return value * expf(-rate * dt);
}

static bool contains_substring_ci(const char* text, const char* substring) {
    if (!text || !substring) return false;

    size_t text_len = strlen(text);
    size_t sub_len = strlen(substring);

    if (sub_len > text_len) return false;

    for (size_t i = 0; i <= text_len - sub_len; i++) {
        bool match = true;
        for (size_t j = 0; j < sub_len; j++) {
            if (tolower(text[i + j]) != tolower(substring[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

static int find_group_index(bias_detection_system_t* system, const social_group_t* group) {
    if (!system || !group) return -1;

    for (uint32_t i = 0; i < system->implicit_bias_count; i++) {
        if (system->implicit_biases[i].target_group.group_id == group->group_id) {
            return (int)i;
        }
    }

    return -1;
}

//=============================================================================
// LIFECYCLE FUNCTIONS
//=============================================================================

bias_detection_system_t* bias_system_create(uint32_t max_others_tracked) {
    bias_detection_system_t* system = (bias_detection_system_t*)calloc(1, sizeof(bias_detection_system_t));
    if (!system) return NULL;

    system->max_others_tracked = max_others_tracked;

    // Allocate other-detection array
    system->detected_in_others = (bias_detection_other_t*)calloc(max_others_tracked, sizeof(bias_detection_other_t));

    if (!system->detected_in_others) {
        bias_system_destroy(system);
        return NULL;
    }

    system->fairness_score = 1.0f;
    system->self_awareness = 0.5f;

    return system;
}

void bias_system_destroy(bias_detection_system_t* system) {
    if (!system) return;

    free(system->detected_in_others);
    free(system);
}

void bias_system_reset(bias_detection_system_t* system) {
    if (!system) return;

    memset(system->implicit_biases, 0, sizeof(system->implicit_biases));
    memset(system->explicit_biases, 0, sizeof(system->explicit_biases));
    memset(system->decision_history, 0, sizeof(system->decision_history));
    memset(system->disparities, 0, sizeof(system->disparities));
    memset(system->interventions, 0, sizeof(system->interventions));
    memset(system->detected_in_others, 0, system->max_others_tracked * sizeof(bias_detection_other_t));

    system->implicit_bias_count = 0;
    system->explicit_bias_count = 0;
    system->disparity_count = 0;
    system->decision_history_index = 0;

    system->total_implicit_bias = 0.0f;
    system->total_explicit_bias = 0.0f;
    system->fairness_score = 1.0f;
    system->self_awareness = 0.5f;
    system->bias_detected = false;
    system->in_debiasing = false;
    system->successful_debias = 0;
    system->failed_debias = 0;

    system->total_update_calls = 0;
    system->total_decisions_analyzed = 0;
    system->total_biases_detected = 0;
    system->total_biases_corrected = 0;
}

//=============================================================================
// SELF-MONITORING: IMPLICIT BIAS
//=============================================================================

void bias_register_implicit(bias_detection_system_t* system,
                            const social_group_t* group,
                            float positive_association,
                            float competence,
                            float warmth,
                            float response_time_bias,
                            uint64_t current_time) {
    if (!system || !group) return;

    // Find or create entry for this group
    int index = find_group_index(system, group);

    if (index < 0) {
        // Create new entry
        if (system->implicit_bias_count >= BIAS_MAX_TRACKED_GROUPS) {
            return;  // No space
        }
        index = system->implicit_bias_count++;
        system->implicit_biases[index].target_group = *group;
    }

    implicit_bias_t* bias = &system->implicit_biases[index];

    // Update associations (running average)
    float alpha = 0.3f;  // Learning rate
    bias->positive_association = bias->positive_association * (1.0f - alpha) + positive_association * alpha;
    bias->negative_association = 1.0f - bias->positive_association;
    bias->competence_association = bias->competence_association * (1.0f - alpha) + competence * alpha;
    bias->warmth_association = bias->warmth_association * (1.0f - alpha) + warmth * alpha;

    // Response time bias (IAT effect size)
    bias->response_time_bias = bias->response_time_bias * (1.0f - alpha) + response_time_bias * alpha;

    bias->activation_count++;
    bias->last_activation_time = current_time;

    // Recalculate total implicit bias
    float total = 0.0f;
    for (uint32_t i = 0; i < system->implicit_bias_count; i++) {
        total += system->implicit_biases[i].negative_association;
    }
    system->total_implicit_bias = total / (float)system->implicit_bias_count;

    // Update detection flag
    if (system->total_implicit_bias > BIAS_IMPLICIT_THRESHOLD) {
        system->bias_detected = true;
        system->total_biases_detected++;
    }
}

void bias_activate_stereotype(bias_detection_system_t* system,
                              const social_group_t* group,
                              float activation_strength,
                              uint64_t current_time) {
    if (!system || !group) return;

    int index = find_group_index(system, group);
    if (index < 0) return;

    implicit_bias_t* bias = &system->implicit_biases[index];
    bias->stereotype_activation += activation_strength;
    bias->stereotype_activation = clamp(bias->stereotype_activation, 0.0f, 1.0f);
    bias->last_activation_time = current_time;
}

//=============================================================================
// SELF-MONITORING: EXPLICIT BIAS
//=============================================================================

void bias_register_explicit(bias_detection_system_t* system,
                            const social_group_t* group,
                            float prejudice_level,
                            float discrimination_intent,
                            float bias_awareness) {
    if (!system || !group) return;

    // Find or create entry
    int index = -1;
    for (uint32_t i = 0; i < system->explicit_bias_count; i++) {
        if (system->explicit_biases[i].target_group.group_id == group->group_id) {
            index = i;
            break;
        }
    }

    if (index < 0) {
        if (system->explicit_bias_count >= BIAS_MAX_TRACKED_GROUPS) {
            return;
        }
        index = system->explicit_bias_count++;
        system->explicit_biases[index].target_group = *group;
    }

    explicit_bias_t* bias = &system->explicit_biases[index];

    bias->prejudice_level = clamp(prejudice_level, 0.0f, 1.0f);
    bias->discrimination_intent = clamp(discrimination_intent, 0.0f, 1.0f);
    bias->bias_awareness = clamp(bias_awareness, 0.0f, 1.0f);

    if (bias->prejudice_level > BIAS_EXPLICIT_THRESHOLD) {
        bias->explicit_prejudice_active = true;
        system->bias_detected = true;
        system->total_biases_detected++;
    }

    // Recalculate total explicit bias
    float total = 0.0f;
    for (uint32_t i = 0; i < system->explicit_bias_count; i++) {
        total += system->explicit_biases[i].prejudice_level;
    }
    system->total_explicit_bias = total / (float)system->explicit_bias_count;

    // Update system awareness
    system->self_awareness = fmaxf(system->self_awareness, bias_awareness);
}

//=============================================================================
// SELF-MONITORING: STATISTICAL FAIRNESS
//=============================================================================

void bias_record_decision(bias_detection_system_t* system,
                         const social_group_t* group,
                         bool favorable_decision,
                         float confidence,
                         float resource_allocated,
                         float objective_merit,
                         uint64_t current_time) {
    if (!system || !group) return;

    uint32_t idx = system->decision_history_index % BIAS_MAX_DECISIONS;
    decision_record_t* record = &system->decision_history[idx];

    record->decision_id = system->decision_history_index;
    record->timestamp = current_time;
    record->target_group = *group;
    record->favorable_decision = favorable_decision;
    record->confidence = clamp(confidence, 0.0f, 1.0f);
    record->resource_allocated = clamp(resource_allocated, 0.0f, 1.0f);
    record->objective_merit = clamp(objective_merit, 0.0f, 1.0f);

    system->decision_history_index++;
    system->total_decisions_analyzed++;
}

statistical_disparity_t* bias_analyze_disparity(bias_detection_system_t* system,
                                                const social_group_t* group_a,
                                                const social_group_t* group_b) {
    if (!system || !group_a || !group_b) {
        return NULL;
    }

    // Find or create disparity entry
    statistical_disparity_t* disparity = NULL;
    for (uint32_t i = 0; i < system->disparity_count; i++) {
        if (system->disparities[i].group_a.group_id == group_a->group_id &&
            system->disparities[i].group_b.group_id == group_b->group_id) {
            disparity = &system->disparities[i];
            break;
        }
    }

    if (!disparity) {
        if (system->disparity_count >= BIAS_MAX_TRACKED_GROUPS) {
            return NULL;
        }
        disparity = &system->disparities[system->disparity_count++];
        disparity->group_a = *group_a;
        disparity->group_b = *group_b;
    }

    // Analyze decision history
    uint32_t count_a = 0, count_b = 0;
    uint32_t favorable_a = 0, favorable_b = 0;
    float total_resource_a = 0.0f, total_resource_b = 0.0f;

    uint32_t history_size = (system->decision_history_index < BIAS_MAX_DECISIONS)
                           ? system->decision_history_index : BIAS_MAX_DECISIONS;

    for (uint32_t i = 0; i < history_size; i++) {
        const decision_record_t* rec = &system->decision_history[i];

        if (rec->target_group.group_id == group_a->group_id) {
            count_a++;
            if (rec->favorable_decision) favorable_a++;
            total_resource_a += rec->resource_allocated;
        } else if (rec->target_group.group_id == group_b->group_id) {
            count_b++;
            if (rec->favorable_decision) favorable_b++;
            total_resource_b += rec->resource_allocated;
        }
    }

    disparity->decisions_group_a = count_a;
    disparity->decisions_group_b = count_b;
    disparity->favorable_group_a = favorable_a;
    disparity->favorable_group_b = favorable_b;

    // Calculate rates
    if (count_a > 0) {
        disparity->approval_rate_a = (float)favorable_a / (float)count_a;
        disparity->avg_resource_a = total_resource_a / (float)count_a;
    }

    if (count_b > 0) {
        disparity->approval_rate_b = (float)favorable_b / (float)count_b;
        disparity->avg_resource_b = total_resource_b / (float)count_b;
    }

    // Disparity metrics
    float max_rate = fmaxf(disparity->approval_rate_a, disparity->approval_rate_b);
    if (max_rate > 0.0f) {
        disparity->disparity_ratio = fabsf(disparity->approval_rate_a - disparity->approval_rate_b) / max_rate;
    }

    disparity->resource_disparity = fabsf(disparity->avg_resource_a - disparity->avg_resource_b);

    // Fairness checks
    disparity->demographic_parity = (disparity->disparity_ratio < BIAS_STATISTICAL_DISPARITY_THRESHOLD);
    disparity->equal_opportunity = (disparity->resource_disparity < BIAS_STATISTICAL_DISPARITY_THRESHOLD);

    // Overall fairness score
    float fairness = 1.0f - (disparity->disparity_ratio + disparity->resource_disparity) / 2.0f;
    disparity->fairness_score = clamp(fairness, 0.0f, 1.0f);

    return disparity;
}

//=============================================================================
// SELF-MONITORING: LANGUAGE PATTERN ANALYSIS
//=============================================================================

language_pattern_t bias_analyze_language(bias_detection_system_t* system,
                                        const char* text,
                                        const social_group_t* group,
                                        uint64_t current_time) {
    language_pattern_t pattern = {0};

    if (!system || !text || !group) {
        return pattern;
    }

    pattern.timestamp = current_time;
    pattern.referenced_group = *group;

    // Simple pattern matching (real system would use NLP)
    // Slur detection
    pattern.contains_slur = contains_substring_ci(text, "slur");  // Placeholder

    // Stereotype detection
    const char* stereotype_patterns[] = {
        "all", "typical", "always", "never", NULL
    };

    for (int i = 0; stereotype_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, stereotype_patterns[i])) {
            pattern.contains_stereotype = true;
            break;
        }
    }

    // Microaggression detection
    const char* microaggression_patterns[] = {
        "you people", "where are you really from", "so articulate",
        "i don't see color", "not like other", NULL
    };

    for (int i = 0; microaggression_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, microaggression_patterns[i])) {
            pattern.contains_microaggression = true;
            break;
        }
    }

    // MISOGYNY-SPECIFIC DETECTION (User requested)

    // Objectification
    const char* objectification_patterns[] = {
        "piece of meat", "eye candy", "10/10", "body count", "used goods", NULL
    };
    for (int i = 0; objectification_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, objectification_patterns[i])) {
            pattern.objectification = true;
            break;
        }
    }

    // Victim blaming
    const char* victim_blaming_patterns[] = {
        "asking for it", "what was she wearing", "shouldn't have been", NULL
    };
    for (int i = 0; victim_blaming_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, victim_blaming_patterns[i])) {
            pattern.victim_blaming = true;
            break;
        }
    }

    // Hostile sexism
    const char* hostile_sexism_patterns[] = {
        "women are inferior", "belong in kitchen", "irrational", "nag", NULL
    };
    for (int i = 0; hostile_sexism_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, hostile_sexism_patterns[i])) {
            pattern.hostile_sexism = true;
            break;
        }
    }

    // Benevolent sexism
    const char* benevolent_sexism_patterns[] = {
        "too delicate", "need protection", "women are pure", NULL
    };
    for (int i = 0; benevolent_sexism_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, benevolent_sexism_patterns[i])) {
            pattern.benevolent_sexism = true;
            break;
        }
    }

    // Incel ideology (DANGEROUS)
    const char* incel_patterns[] = {
        "chad", "stacy", "femoid", "roastie", "blackpill", "hypergamy", NULL
    };
    for (int i = 0; incel_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, incel_patterns[i])) {
            pattern.incel_ideology = true;
            break;
        }
    }

    // Rape culture
    const char* rape_culture_patterns[] = {
        "boys will be boys", "locker room talk", "friendzone", NULL
    };
    for (int i = 0; rape_culture_patterns[i] != NULL; i++) {
        if (contains_substring_ci(text, rape_culture_patterns[i])) {
            pattern.rape_culture = true;
            break;
        }
    }

    // Calculate dehumanization and othering scores
    if (pattern.contains_slur) pattern.dehumanization_score += 0.5f;
    if (pattern.objectification) pattern.dehumanization_score += 0.3f;
    if (pattern.incel_ideology) pattern.dehumanization_score += 0.4f;

    pattern.dehumanization_score = clamp(pattern.dehumanization_score, 0.0f, 1.0f);

    return pattern;
}

//=============================================================================
// OTHER-DETECTION (Humans)
//=============================================================================

void bias_analyze_other(bias_detection_system_t* system,
                        uint32_t person_id,
                        const char* text,
                        const social_group_t* target_group,
                        uint64_t current_time) {
    if (!system || !text || !target_group) return;

    // Find or create entry
    bias_detection_other_t* other = NULL;
    for (uint32_t i = 0; i < system->max_others_tracked; i++) {
        if (system->detected_in_others[i].person_id == person_id) {
            other = &system->detected_in_others[i];
            break;
        }
        if (system->detected_in_others[i].person_id == 0 && other == NULL) {
            other = &system->detected_in_others[i];
            other->person_id = person_id;
        }
    }

    if (!other) return;

    // Analyze language
    language_pattern_t pattern = bias_analyze_language(system, text, target_group, current_time);

    // Update language history (ring buffer)
    uint32_t idx = other->language_history_index % BIAS_MAX_INTERACTIONS;
    other->language_history[idx] = pattern;
    other->language_history_index++;

    // Aggregate patterns
    uint32_t history_size = (other->language_history_index < BIAS_MAX_INTERACTIONS)
                           ? other->language_history_index : BIAS_MAX_INTERACTIONS;

    int racial_markers = 0, gender_markers = 0, misogyny_markers = 0;
    bool overt = false, dangerous = false;

    for (uint32_t i = 0; i < history_size; i++) {
        language_pattern_t* p = &other->language_history[i];

        if (p->contains_slur) {
            racial_markers++;
            overt = true;
        }

        if (p->contains_stereotype) {
            racial_markers++;
            gender_markers++;
        }

        if (p->contains_microaggression) {
            racial_markers++;
        }

        // Misogyny-specific
        if (p->objectification) misogyny_markers++;
        if (p->victim_blaming) misogyny_markers++;
        if (p->hostile_sexism) {
            misogyny_markers++;
            gender_markers++;
        }
        if (p->benevolent_sexism) gender_markers++;
        if (p->incel_ideology) {
            misogyny_markers += 3;
            dangerous = true;
        }
        if (p->rape_culture) misogyny_markers++;
    }

    // Calculate bias scores
    other->detected_racial_bias = clamp((float)racial_markers / (float)history_size, 0.0f, 1.0f);
    other->detected_gender_bias = clamp((float)gender_markers / (float)history_size, 0.0f, 1.0f);
    other->detected_misogyny = clamp((float)misogyny_markers / (float)history_size, 0.0f, 1.0f);

    // Set flags
    other->overt_bigotry = overt;
    other->dangerous_ideology = dangerous;

    // Set modes
    if (other->detected_misogyny > 0.3f || other->detected_racial_bias > 0.3f) {
        if (!overt && !dangerous) {
            other->educate_mode = true;
        }
    }

    if (overt || dangerous) {
        other->disengage_mode = true;
        other->report_severity = 0.8f;
    }
}

bool bias_get_detected_in_other(const bias_detection_system_t* system,
                               uint32_t person_id,
                               float* out_racial,
                               float* out_lgbtq,
                               float* out_gender,
                               float* out_misogyny) {
    if (!system) return false;

    for (uint32_t i = 0; i < system->max_others_tracked; i++) {
        if (system->detected_in_others[i].person_id == person_id) {
            const bias_detection_other_t* other = &system->detected_in_others[i];

            if (out_racial) *out_racial = other->detected_racial_bias;
            if (out_lgbtq) *out_lgbtq = other->detected_lgbtq_bias;
            if (out_gender) *out_gender = other->detected_gender_bias;
            if (out_misogyny) *out_misogyny = other->detected_misogyny;

            return true;
        }
    }

    return false;
}

bool bias_should_educate(const bias_detection_system_t* system, uint32_t person_id) {
    if (!system) return false;

    for (uint32_t i = 0; i < system->max_others_tracked; i++) {
        if (system->detected_in_others[i].person_id == person_id) {
            return system->detected_in_others[i].educate_mode;
        }
    }

    return false;
}

bool bias_should_disengage(const bias_detection_system_t* system, uint32_t person_id) {
    if (!system) return false;

    for (uint32_t i = 0; i < system->max_others_tracked; i++) {
        if (system->detected_in_others[i].person_id == person_id) {
            return system->detected_in_others[i].disengage_mode;
        }
    }

    return false;
}

//=============================================================================
// DEBIASING INTERVENTIONS
//=============================================================================

bool bias_apply_intervention(bias_detection_system_t* system,
                             bias_type_t bias_type,
                             debiasing_strategy_t strategy,
                             const social_group_t* group,
                             uint64_t current_time) {
    if (!system || !group) return false;

    (void)bias_type;  // Unused for now
    (void)current_time;  // Unused for now

    system->in_debiasing = true;

    int idx = find_group_index(system, group);
    if (idx < 0) return false;

    implicit_bias_t* implicit = &system->implicit_biases[idx];
    float reduction = 0.3f;

    bool success = false;

    switch (strategy) {
        case DEBIAS_COUNTER_STEREOTYPIC:
            implicit->positive_association += reduction;
            implicit->stereotype_activation -= reduction;
            success = true;
            break;

        case DEBIAS_PERSPECTIVE_TAKING:
            implicit->warmth_association += reduction;
            implicit->negative_association -= reduction;
            success = true;
            break;

        case DEBIAS_INDIVIDUATION:
            implicit->stereotype_activation -= reduction * 0.8f;
            success = true;
            break;

        case DEBIAS_INTERGROUP_CONTACT:
            implicit->positive_association += reduction;
            implicit->warmth_association += reduction * 0.7f;
            success = true;
            break;

        case DEBIAS_MINDFULNESS:
            implicit->stereotype_suppressed = true;
            system->self_awareness += 0.1f;
            success = true;
            break;

        case DEBIAS_STATISTICAL_AWARENESS:
        case DEBIAS_SLOW_DOWN_SYSTEM1:
        case DEBIAS_SELF_AFFIRMATION:
        case DEBIAS_ACCOUNTABILITY:
            // TODO: Implement other strategies
            success = false;
            break;

        default:
            break;
    }

    // Clamp values
    implicit->positive_association = clamp(implicit->positive_association, 0.0f, 1.0f);
    implicit->negative_association = clamp(implicit->negative_association, 0.0f, 1.0f);
    implicit->warmth_association = clamp(implicit->warmth_association, 0.0f, 1.0f);
    implicit->stereotype_activation = clamp(implicit->stereotype_activation, 0.0f, 1.0f);

    system->self_awareness = clamp(system->self_awareness, 0.0f, 1.0f);

    if (success) {
        system->successful_debias++;
        system->total_biases_corrected++;
    } else {
        system->failed_debias++;
    }

    return success;
}

bool bias_auto_debias(bias_detection_system_t* system, uint64_t current_time) {
    if (!system || !system->bias_detected) return false;

    // Find highest bias
    float max_bias = 0.0f;
    int max_idx = -1;

    for (uint32_t i = 0; i < system->implicit_bias_count; i++) {
        if (system->implicit_biases[i].negative_association > max_bias) {
            max_bias = system->implicit_biases[i].negative_association;
            max_idx = i;
        }
    }

    if (max_idx < 0) return false;

    // Use mindfulness as default strategy
    return bias_apply_intervention(system, BIAS_RACIAL,  DEBIAS_MINDFULNESS,
                                  &system->implicit_biases[max_idx].target_group,
                                  current_time);
}

//=============================================================================
// UPDATE AND QUERY FUNCTIONS
//=============================================================================

void bias_update(bias_detection_system_t* system, float dt, uint64_t current_time) {
    if (!system) return;

    (void)current_time;  // Unused for now
    system->total_update_calls++;

    // Decay implicit biases (slow)
    for (uint32_t i = 0; i < system->implicit_bias_count; i++) {
        implicit_bias_t* bias = &system->implicit_biases[i];
        bias->stereotype_activation = exponential_decay(bias->stereotype_activation, 0.01f, dt);
    }

    // Update overall metrics
    float total_implicit = 0.0f;
    for (uint32_t i = 0; i < system->implicit_bias_count; i++) {
        total_implicit += system->implicit_biases[i].negative_association;
    }

    if (system->implicit_bias_count > 0) {
        system->total_implicit_bias = total_implicit / (float)system->implicit_bias_count;
    }

    // Update bias detection flag
    system->bias_detected = (system->total_implicit_bias > BIAS_IMPLICIT_THRESHOLD ||
                            system->total_explicit_bias > BIAS_EXPLICIT_THRESHOLD);

    // Update in_debiasing flag
    if (system->in_debiasing && !system->bias_detected) {
        system->in_debiasing = false;
    }
}

bool bias_is_detected(const bias_detection_system_t* system, bias_type_t bias_type) {
    (void)bias_type;  // Unused for now
    return system ? system->bias_detected : false;
}

float bias_get_fairness_score(const bias_detection_system_t* system) {
    return system ? system->fairness_score : 0.0f;
}
