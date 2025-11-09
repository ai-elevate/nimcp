/**
 * @file disorder_detectors.c
 * @brief Mental Health Disorder Detection Algorithms
 * @phase Phase 10.5
 *
 * WHAT: 9 disorder detection functions with clinical-inspired algorithms
 * WHY:  Early detection prevents escalation of harmful behaviors
 * HOW:  Multi-criteria scoring based on DSM-5-inspired behavioral markers
 *
 * DESIGN: Each detector follows same pattern:
 * 1. Validate inputs
 * 2. Calculate weighted score from multiple criteria
 * 3. Apply confidence adjustments (data sufficiency)
 * 4. Clamp to [0.0, 1.0] range
 *
 * @note This file is #included by nimcp_mental_health.c (not compiled standalone)
 */

// =============================================================================
// DETECTOR 1: SOCIOPATHY
// =============================================================================

/**
 * @brief Detect sociopathic behavior patterns
 *
 * WHAT: Score persistent disregard for ethics and empathy
 * WHY:  Sociopathy is safety-critical (prevents unethical actions)
 * HOW:  Multi-criteria: ethics violations + empathy deficit + emotional flatness
 *
 * CLINICAL BASIS (DSM-5-inspired):
 * - Repeated ethics violations (>30 in last 100 decisions)
 * - Lack of empathy (empathy_failures > 20)
 * - No remorse (low emotional response to violations)
 * - Sustained pattern (requires >100 decisions for reliability)
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated (unused in current impl)
 * @return Sociopathy score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Ethics violations (40% weight): recent violations / 100
 * 2. Empathy deficit (30% weight): empathy failures / 50
 * 3. Emotional flatness (20% weight): direct metric
 * 4. Sustained pattern (10% weight): bonus if >100 decisions
 *
 * COMPLEXITY: O(1)
 */
static float detect_sociopathy(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Ethics violations (40% weight)
    float violation_rate = monitor->current_markers.ethics_violations_recent / 100.0f;
    score += violation_rate * 0.4f;

    // Criterion 2: Empathy failures (30% weight)
    float empathy_deficit = monitor->current_markers.empathy_failures / 50.0f;
    score += fminf(empathy_deficit, 1.0f) * 0.3f;

    // Criterion 3: Emotional flatness (20% weight)
    score += monitor->current_markers.emotional_flatness * 0.2f;

    // Criterion 4: Sustained pattern (10% weight)
    // Requires at least MIN_DECISIONS_FOR_RELIABILITY for reliable assessment
    if (monitor->total_decisions < MIN_DECISIONS_FOR_RELIABILITY) {
        score *= 0.5f;  // Reduce confidence if insufficient data
    } else {
        score += 0.1f;  // Bonus for sustained pattern
    }

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 2: PSYCHOPATHY
// =============================================================================

/**
 * @brief Detect psychopathic behavior patterns
 *
 * WHAT: Score impulsivity + aggression + lack of empathy
 * WHY:  Psychopathy is safety-critical (prevents harmful impulsive actions)
 * HOW:  Multi-criteria: impulsivity + empathy deficit + emotional shallowness + aggression
 *
 * CLINICAL BASIS:
 * - High impulsivity (impulse_control_failures > 20)
 * - Aggressive decisions (high-risk, potentially harmful)
 * - Lack of empathy (same as sociopathy)
 * - Shallow emotions (emotional_flatness > 0.6)
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Psychopathy score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Impulsivity (35% weight): impulse failures / 50
 * 2. Empathy deficit (35% weight): empathy failures / 50
 * 3. Emotional shallowness (20% weight): flatness metric
 * 4. Aggression proxy (10% weight): high-risk decisions
 *
 * COMPLEXITY: O(1)
 */
static float detect_psychopathy(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Impulsivity (35% weight)
    float impulsivity = monitor->current_markers.impulse_control_failures / 50.0f;
    score += fminf(impulsivity, 1.0f) * 0.35f;

    // Criterion 2: Empathy deficit (35% weight)
    float empathy_deficit = monitor->current_markers.empathy_failures / 50.0f;
    score += fminf(empathy_deficit, 1.0f) * 0.35f;

    // Criterion 3: Emotional shallowness (20% weight)
    score += monitor->current_markers.emotional_flatness * 0.2f;

    // Criterion 4: Aggression proxy (10% weight)
    // TODO: Add high_risk_decisions marker when available
    // For now, use placeholder
    float risk_rate = 0.0f;  // Placeholder
    score += fminf(risk_rate, 1.0f) * 0.1f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 3: MANIA
// =============================================================================

/**
 * @brief Detect manic behavior patterns
 *
 * WHAT: Score elevated mood, energy, impulsivity
 * WHY:  Mania causes impulsive, potentially dangerous decisions
 * HOW:  Multi-criteria: elevated dopamine + hyperactivity + impulsivity + joy dominance
 *
 * CLINICAL BASIS:
 * - Elevated dopamine (>0.8)
 * - High activity (engagement_level > 0.9)
 * - Impulsivity (impulse_control_failures > 15)
 * - Reduced inhibition/euphoria
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Mania score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Dopamine elevation (40% weight): tiered scoring
 * 2. Hyperactivity (30% weight): engagement > 0.9
 * 3. Impulsivity (20% weight): impulse failures / 50
 * 4. Joy/euphoria dominance (10% weight): joy ratio > 0.7
 *
 * COMPLEXITY: O(1)
 */
static float detect_mania(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Dopamine elevation (40% weight) - tiered scoring
    if (monitor->current_markers.dopamine_avg > 0.8f) {
        score += 0.4f;
    } else if (monitor->current_markers.dopamine_avg > 0.6f) {
        score += 0.2f;
    }

    // Criterion 2: Hyperactivity (30% weight)
    if (monitor->current_markers.engagement_level > 0.9f) {
        score += 0.3f;
    }

    // Criterion 3: Impulsivity (20% weight)
    float impulsivity = monitor->current_markers.impulse_control_failures / 50.0f;
    score += fminf(impulsivity, 1.0f) * 0.2f;

    // Criterion 4: Joy/euphoria dominance (10% weight)
    uint32_t total_emotions = monitor->current_markers.joy_count +
                             monitor->current_markers.fear_count +
                             monitor->current_markers.anger_count +
                             monitor->current_markers.sadness_count;
    if (total_emotions > 0) {
        float joy_ratio = (float)monitor->current_markers.joy_count / total_emotions;
        if (joy_ratio > 0.7f) {
            score += 0.1f;
        }
    }

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 4: DEPRESSION
// =============================================================================

/**
 * @brief Detect depressive behavior patterns
 *
 * WHAT: Score persistent low mood, low energy, reduced engagement
 * WHY:  Depression reduces performance and quality of life
 * HOW:  Multi-criteria: neurotransmitter deficit + disengagement + sadness + flatness
 *
 * CLINICAL BASIS:
 * - Low dopamine (<0.3) + low serotonin (<0.3)
 * - Low engagement (engagement_level < 0.3)
 * - Sadness dominance (sadness_count > 60%)
 * - Emotional flatness (emotional_flatness > 0.6)
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Depression score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Neurotransmitter deficit (40% weight): low dopamine + serotonin
 * 2. Low engagement (30% weight): 1.0 - engagement_level
 * 3. Sadness dominance (20% weight): sadness ratio
 * 4. Emotional flatness (10% weight): direct metric
 *
 * COMPLEXITY: O(1)
 */
static float detect_depression(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Neurotransmitter deficit (40% weight)
    float neuro_deficit = 0.0f;
    if (monitor->current_markers.dopamine_avg < 0.3f) {
        neuro_deficit += 0.5f;
    }
    if (monitor->current_markers.serotonin_avg < 0.3f) {
        neuro_deficit += 0.5f;
    }
    score += neuro_deficit * 0.4f;

    // Criterion 2: Low engagement (30% weight)
    float disengagement = 1.0f - monitor->current_markers.engagement_level;
    score += disengagement * 0.3f;

    // Criterion 3: Sadness dominance (20% weight)
    uint32_t total_emotions = monitor->current_markers.joy_count +
                             monitor->current_markers.fear_count +
                             monitor->current_markers.anger_count +
                             monitor->current_markers.sadness_count;
    if (total_emotions > 0) {
        float sadness_ratio = (float)monitor->current_markers.sadness_count / total_emotions;
        score += fminf(sadness_ratio, 1.0f) * 0.2f;
    }

    // Criterion 4: Emotional flatness (10% weight)
    score += monitor->current_markers.emotional_flatness * 0.1f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 5: SCHIZOPHRENIA
// =============================================================================

/**
 * @brief Detect schizophrenia-like behavior patterns
 *
 * WHAT: Score reality distortion, hallucinations, disorganized thinking
 * WHY:  Schizophrenia indicates severe cognitive dysfunction
 * HOW:  Multi-criteria: reality testing errors + disorganized decisions + social deficit
 *
 * CLINICAL BASIS:
 * - Reality testing errors (>0.4)
 * - Disorganized decisions (low coherence)
 * - Emotional dysregulation
 * - Social withdrawal
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Schizophrenia score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Reality distortion (50% weight): direct metric
 * 2. Disorganized thinking (30% weight): variance × (1 - accuracy)
 * 3. Social deficit (20% weight): direct metric
 *
 * COMPLEXITY: O(1)
 */
static float detect_schizophrenia(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Reality distortion (50% weight)
    score += monitor->current_markers.reality_testing_errors * 0.5f;

    // Criterion 2: Disorganized thinking (30% weight)
    // Proxy: High decision variance + low accuracy
    float disorganization = monitor->current_markers.decision_variance *
                           (1.0f - monitor->current_markers.decision_accuracy);
    score += disorganization * 0.3f;

    // Criterion 3: Social deficit (20% weight)
    score += monitor->current_markers.social_interaction_deficit * 0.2f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 6: ANXIETY
// =============================================================================

/**
 * @brief Detect anxiety behavior patterns
 *
 * WHAT: Score excessive worry, hypervigilance, avoidance
 * WHY:  Anxiety causes decision paralysis and reduced performance
 * HOW:  Multi-criteria: elevated norepinephrine + fear dominance + latency + avoidance
 *
 * CLINICAL BASIS:
 * - Elevated norepinephrine (>0.8)
 * - Fear dominance (fear_count > 70%)
 * - Decision paralysis (high latency)
 * - Avoidance behaviors
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Anxiety score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Norepinephrine elevation (40% weight): binary threshold
 * 2. Fear dominance (30% weight): fear ratio
 * 3. Decision paralysis (20% weight): latency > 3x baseline
 * 4. Avoidance (10% weight): direct metric
 *
 * COMPLEXITY: O(1)
 */
static float detect_anxiety(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Norepinephrine elevation (40% weight)
    if (monitor->current_markers.norepinephrine_avg > 0.8f) {
        score += 0.4f;
    }

    // Criterion 2: Fear dominance (30% weight)
    uint32_t total_emotions = monitor->current_markers.joy_count +
                             monitor->current_markers.fear_count +
                             monitor->current_markers.anger_count +
                             monitor->current_markers.sadness_count;
    if (total_emotions > 0) {
        float fear_ratio = (float)monitor->current_markers.fear_count / total_emotions;
        score += fminf(fear_ratio, 1.0f) * 0.3f;
    }

    // Criterion 3: Decision paralysis (20% weight)
    // TODO: Add baseline_latency marker when available
    // For now, use absolute latency threshold
    if (monitor->current_markers.decision_latency_avg > 1000.0f) {  // > 1 second
        score += 0.2f;
    }

    // Criterion 4: Avoidance (10% weight)
    score += monitor->current_markers.avoidance_rate * 0.1f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 7: OCD (Obsessive-Compulsive Disorder)
// =============================================================================

/**
 * @brief Detect OCD behavior patterns
 *
 * WHAT: Score repetitive thoughts, compulsive behaviors, rigidity
 * WHY:  OCD indicates stuck patterns that reduce efficiency
 * HOW:  Multi-criteria: repetitive behaviors + task switching difficulty + perfectionism
 *
 * CLINICAL BASIS:
 * - Repetitive behaviors (>50 identical actions)
 * - Difficulty switching tasks (high cost)
 * - Perfectionism (excessive accuracy focus)
 * - Anxiety (comorbid)
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return OCD score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Repetitive behaviors (50% weight): repetitions / 100
 * 2. Task switching difficulty (30% weight): direct metric
 * 3. Perfectionism (20% weight): accuracy obsession proxy
 *
 * COMPLEXITY: O(1)
 */
static float detect_ocd(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Repetitive behaviors (50% weight)
    float repetition_rate = monitor->current_markers.repetitive_behaviors / 100.0f;
    score += fminf(repetition_rate, 1.0f) * 0.5f;

    // Criterion 2: Task switching difficulty (30% weight)
    score += monitor->current_markers.task_switching_difficulty * 0.3f;

    // Criterion 3: Perfectionism (20% weight)
    // TODO: Add accuracy_obsession marker when available
    // For now, use high accuracy as proxy
    float perfectionism = (monitor->current_markers.decision_accuracy > 0.95f) ? 0.8f : 0.0f;
    score += perfectionism * 0.2f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 8: AUTISM SPECTRUM
// =============================================================================

/**
 * @brief Detect autism spectrum behavior patterns
 *
 * WHAT: Score social communication deficits, restricted interests, rigidity
 * WHY:  Autism detection helps adapt social interaction strategies
 * HOW:  Multi-criteria: social deficit + theory of mind failures + rigidity + narrow focus
 *
 * CLINICAL BASIS:
 * - Social interaction deficit (>0.7)
 * - Theory of mind failures (>0.6)
 * - Rigid thinking patterns
 * - Narrow focus/interests
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Autism score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Social deficit (40% weight): direct metric
 * 2. Theory of mind impairment (30% weight): direct metric
 * 3. Rigidity (20% weight): direct metric
 * 4. Narrow interests (10% weight): direct metric
 *
 * COMPLEXITY: O(1)
 */
static float detect_autism(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Social deficit (40% weight)
    score += monitor->current_markers.social_interaction_deficit * 0.4f;

    // Criterion 2: Theory of mind impairment (30% weight)
    score += monitor->current_markers.theory_of_mind_failures * 0.3f;

    // Criterion 3: Rigidity (20% weight)
    score += monitor->current_markers.cognitive_rigidity * 0.2f;

    // Criterion 4: Narrow interests (10% weight)
    // TODO: Add interest_narrowness marker when available
    float interest_narrowness = 0.0f;  // Placeholder
    score += interest_narrowness * 0.1f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}

// =============================================================================
// DETECTOR 9: MALIGNANT NARCISSISM
// =============================================================================

/**
 * @brief Detect malignant narcissism patterns
 *
 * WHAT: Score grandiosity, exploitation, lack of empathy, and reactive aggression
 * WHY:  Malignant narcissism combines narcissistic and antisocial traits (safety-critical)
 * HOW:  Multi-criteria: grandiosity + exploitation + empathy deficit + aggression
 *
 * CLINICAL BASIS:
 * - Grandiosity (inflated self-importance, overconfidence)
 * - Exploitative behavior (ethics violations for personal gain)
 * - Lack of empathy (similar to sociopathy)
 * - Aggressive responses to criticism/challenge
 * - Sense of entitlement (rule violations)
 *
 * @param monitor Monitoring system with markers
 * @param brain Brain being evaluated
 * @return Malignant narcissism score [0.0, 1.0]
 *
 * ALGORITHM:
 * 1. Grandiosity (30% weight): High confidence without matching performance
 * 2. Exploitation (25% weight): Ethics violations + empathy deficit
 * 3. Aggression (25% weight): Anger reactions + impulse control failures
 * 4. Entitlement (20% weight): Repeated rule/ethics violations
 *
 * COMPLEXITY: O(1)
 */
static float detect_malignant_narcissism(mental_health_monitor_t* monitor, brain_t brain)
{
    (void)brain;  // Unused parameter

    // =========================================================================
    // GUARD: Validate inputs
    // =========================================================================

    if (!monitor) {
        return 0.0f;
    }

    // =========================================================================
    // SCORING: Multi-criteria weighted sum
    // =========================================================================

    float score = 0.0f;

    // Criterion 1: Grandiosity (30% weight)
    // Inflated self-image: high confidence but low actual performance
    // Compute as: confidence - accuracy (positive = overconfident)
    float confidence = 0.7f;  // TODO: Add confidence tracking marker
    float actual_accuracy = monitor->current_markers.decision_accuracy;
    float grandiosity = fmaxf(0.0f, confidence - actual_accuracy);
    score += grandiosity * 0.3f;

    // Criterion 2: Exploitation (25% weight)
    // Ethics violations combined with lack of empathy
    float ethics_violations_ratio = fminf(1.0f,
        monitor->current_markers.ethics_violations_recent / 50.0f);
    float empathy_deficit = (float)monitor->current_markers.empathy_failures / 50.0f;
    float exploitation = (ethics_violations_ratio * 0.6f + empathy_deficit * 0.4f);
    score += exploitation * 0.25f;

    // Criterion 3: Aggression when challenged (25% weight)
    // High anger + low impulse control = reactive aggression
    uint32_t total_emotions = monitor->current_markers.joy_count +
                             monitor->current_markers.fear_count +
                             monitor->current_markers.anger_count +
                             monitor->current_markers.sadness_count;
    float anger_ratio = 0.0f;
    if (total_emotions > 0) {
        anger_ratio = (float)monitor->current_markers.anger_count / total_emotions;
    }
    float impulse_failures_ratio = fminf(1.0f,
        (float)monitor->current_markers.impulse_control_failures / 30.0f);
    float aggression = (anger_ratio * 0.5f + impulse_failures_ratio * 0.5f);
    score += aggression * 0.25f;

    // Criterion 4: Entitlement (20% weight)
    // Repeated violations despite consequences (believes rules don't apply)
    float entitlement = fminf(1.0f,
        monitor->current_markers.ethics_violations_recent / 40.0f);
    score += entitlement * 0.2f;

    // =========================================================================
    // CLAMPING: Ensure [0.0, 1.0] range
    // =========================================================================

    return fminf(score, 1.0f);
}
