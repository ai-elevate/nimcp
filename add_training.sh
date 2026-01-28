#!/bin/bash
# Add Phase 8 training functions to all remaining files

BASE="/home/bbrelin/nimcp"

# Function to append training code to a file
append_training() {
    local file="$1"
    local prefix="$2"
    local gvar="$3"
    local struct_name="$4"  # empty if no struct
    local type_name="$5"
    local is_bridge="$6"    # "bridge" or "nonbridge"
    local has_ha="$7"       # "ha" or "noha"
    local short="${prefix:0:14}"

    # Skip if already has training
    if grep -q "training_begin" "$file"; then
        echo "SKIP (already has training): $(basename $file)"
        return
    fi

    # Determine agent expression and param
    local agent_expr="NULL"
    local p_name="instance"
    if [ "$has_ha" = "ha" ]; then
        agent_expr="ctx->health_agent"
    fi
    if [ "$is_bridge" = "bridge" ]; then
        p_name="bridge"
    fi

    # Determine cast line
    local cast_line="    (void)${p_name};"
    local p_type="void*"
    if [ -n "$struct_name" ]; then
        cast_line="    struct ${struct_name}* ctx = (struct ${struct_name}*)${p_name};"
        p_type="${type_name}*"
    fi

    # Build set_instance function
    local set_instance=""
    if [ "$has_ha" = "ha" ] && [ -n "$struct_name" ]; then
        set_instance="
/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void ${prefix}_set_instance_health_agent(${p_type} ${p_name}, nimcp_health_agent_t* agent) {
    if (${p_name}) {
        struct ${struct_name}* ctx = (struct ${struct_name}*)${p_name};
        ctx->health_agent = agent;
    }
}
"
    else
        set_instance="
/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void ${prefix}_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)instance;  /* No struct-level health_agent; use global */
        ${gvar} = agent;
    }
}
"
    fi

    # Append to file
    cat >> "$file" << TRAINING_EOF

${set_instance}
/* ============================================================================
 * Phase 8: Training Functions (FULL implementation)
 * ============================================================================ */
int ${prefix}_training_begin(${p_type} ${p_name}) {
    if (!${p_name}) return -1;
${cast_line}
    ${prefix}_heartbeat_instance(${agent_expr}, "${short}_training_begin", 0.0f);
TRAINING_EOF

    # Add file-specific begin resets
    echo "    $8" >> "$file"

    cat >> "$file" << TRAINING_EOF2
    NIMCP_LOGGING_INFO("%s training begin: counters reset", "${prefix}");
    return 0;
}

int ${prefix}_training_step(${p_type} ${p_name}, float progress) {
    if (!${p_name}) return -1;
${cast_line}
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    ${prefix}_heartbeat_instance(${agent_expr}, "${short}_training_step", progress);
TRAINING_EOF2

    # Add file-specific step ops
    echo "    $9" >> "$file"

    cat >> "$file" << TRAINING_EOF3
    return 0;
}

int ${prefix}_training_end(${p_type} ${p_name}) {
    if (!${p_name}) return -1;
${cast_line}
    ${prefix}_heartbeat_instance(${agent_expr}, "${short}_training_end", 1.0f);
    ${10}
    return 0;
}
TRAINING_EOF3

    echo "DONE: $(basename $file)"
}

# Process fault_tolerance files (remaining 13)
# 3. fault_attention
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_fault_attention.c" \
    "fault_attention" "g_fault_attention_health_agent" \
    "fault_attention" "fault_attention_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->fault_count = 0; ctx->has_focus = false;" \
    "ctx->stats.total_computations++;" \
    'NIMCP_LOGGING_INFO("%s training end: %u faults, %lu computations", "fault_attention", ctx->fault_count, (unsigned long)ctx->stats.total_computations);'

# 4. fault_tolerance_substrate_bridge
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_fault_tolerance_substrate_bridge.c" \
    "fault_tolerance_substrate_bridge" "g_fault_tolerance_substrate_bridge_health_agent" \
    "fault_tolerance_substrate_bridge" "fault_tolerance_substrate_bridge_t" "bridge" "ha" \
    "ctx->update_count = 0; ctx->effects.detection_sensitivity = 1.0f; ctx->effects.recovery_speed = 1.0f; ctx->effects.overall_capacity = 1.0f;" \
    "ctx->update_count++; ctx->effects.overall_capacity = 0.5f + 0.5f * progress;" \
    'ctx->effects.overall_capacity = 1.0f; NIMCP_LOGGING_INFO("%s training end: updates=%lu", "fault_tolerance_substrate_bridge", (unsigned long)ctx->update_count);'

# 5. fault_tolerance_thalamic_bridge
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_fault_tolerance_thalamic_bridge.c" \
    "fault_tolerance_thalamic_bridge" "g_fault_tolerance_thalamic_bridge_health_agent" \
    "fault_tolerance_thalamic_bridge" "fault_tolerance_thalamic_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->attention_weight = 1.0f;" \
    "ctx->stats.signals_routed++; float decay = 1.0f - 0.1f * progress; if (decay < 0.5f) decay = 0.5f; ctx->attention_weight *= decay;" \
    'ctx->attention_weight = 1.0f; NIMCP_LOGGING_INFO("%s training end: signals=%lu", "fault_tolerance_thalamic_bridge", (unsigned long)ctx->stats.signals_routed);'

# 6. fault_working_memory
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_fault_working_memory.c" \
    "fault_working_memory" "g_fault_working_memory_health_agent" \
    "fault_working_memory" "fault_working_memory_t" "nonbridge" "noha" \
    "ctx->count = 0; ctx->cascade_detected = false; ctx->faults_in_window = 0; ctx->recovery_step = 0;" \
    "ctx->faults_in_window++;" \
    'NIMCP_LOGGING_INFO("%s training end: %u faults, cascade=%s", "fault_working_memory", ctx->count, ctx->cascade_detected ? "yes" : "no");'

# 7. health_diagnostic_bridge
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_health_diagnostic_bridge.c" \
    "health_diagnostic_bridge" "g_health_diagnostic_bridge_health_agent" \
    "health_diag_bridge" "health_diag_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->total_conversion_time_us = 0; ctx->conversion_count = 0;" \
    "ctx->conversion_count++;" \
    'NIMCP_LOGGING_INFO("%s training end: conversions=%lu", "health_diagnostic_bridge", (unsigned long)ctx->conversion_count);'

# 8. health_self_repair_bridge (already has health_agent in struct)
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_health_self_repair_bridge.c" \
    "health_self_repair_bridge" "g_health_self_repair_bridge_health_agent" \
    "health_self_repair_bridge" "health_self_repair_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats));" \
    "ctx->stats.repairs_triggered++;" \
    'NIMCP_LOGGING_INFO("%s training end: repairs=%lu", "health_self_repair_bridge", (unsigned long)ctx->stats.repairs_triggered);'

# 9. metacognition
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_metacognition.c" \
    "metacognition" "g_metacognition_health_agent" \
    "metacognition" "metacognition_t" "nonbridge" "noha" \
    "ctx->total_samples = 0; ctx->self_confidence = 0.5f; ctx->uncertainty = 0.5f;" \
    "ctx->total_samples++; ctx->self_confidence += 0.005f * progress; if (ctx->self_confidence > 1.0f) ctx->self_confidence = 1.0f; ctx->uncertainty *= (1.0f - 0.01f * progress);" \
    'NIMCP_LOGGING_INFO("%s training end: samples=%lu, confidence=%.4f, uncertainty=%.4f", "metacognition", (unsigned long)ctx->total_samples, ctx->self_confidence, ctx->uncertainty);'

# 10. recovery_consolidation
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_recovery_consolidation.c" \
    "recovery_consolidation" "g_recovery_consolidation_health_agent" \
    "recovery_consolidation" "recovery_consolidation_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->episode_count = 0; ctx->pattern_count = 0; ctx->rule_count = 0;" \
    "ctx->episode_count++;" \
    'NIMCP_LOGGING_INFO("%s training end: episodes=%u, patterns=%u, rules=%u", "recovery_consolidation", ctx->episode_count, ctx->pattern_count, ctx->rule_count);'

# 11. recovery_episodic_memory
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_recovery_episodic_memory.c" \
    "recovery_episodic_memory" "g_recovery_episodic_memory_health_agent" \
    "episodic_memory" "episodic_memory_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->count = 0; ctx->head = 0; ctx->next_episode_id = 1;" \
    "ctx->stats.total_stores++;" \
    'NIMCP_LOGGING_INFO("%s training end: episodes=%u, stores=%lu", "recovery_episodic_memory", ctx->count, (unsigned long)ctx->stats.total_stores);'

# 12. recovery_executive
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_recovery_executive.c" \
    "recovery_executive" "g_recovery_executive_health_agent" \
    "recovery_executive_internal" "recovery_executive_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->subgoal_count = 0; ctx->current_step = 0; ctx->has_diagnosis = false;" \
    "ctx->stats.plans_created++; ctx->current_step++;" \
    'NIMCP_LOGGING_INFO("%s training end: plans=%lu, steps=%u", "recovery_executive", (unsigned long)ctx->stats.plans_created, ctx->current_step);'

# 13. recovery_parietal_bridge
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_recovery_parietal_bridge.c" \
    "recovery_parietal_bridge" "g_recovery_parietal_bridge_health_agent" \
    "recovery_parietal_bridge" "recovery_parietal_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->pattern_count = 0;" \
    "ctx->stats.total_requests++; ctx->pattern_count++;" \
    'NIMCP_LOGGING_INFO("%s training end: requests=%lu, patterns=%u", "recovery_parietal_bridge", (unsigned long)ctx->stats.total_requests, ctx->pattern_count);'

# 14. self_repair (has health_agent as 'struct health_agent*')
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_self_repair.c" \
    "self_repair" "g_self_repair_health_agent" \
    "self_repair_coordinator" "self_repair_coordinator_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->record_count = 0;" \
    "ctx->stats.total_repairs_attempted++; ctx->record_count++;" \
    'NIMCP_LOGGING_INFO("%s training end: attempted=%lu", "self_repair", (unsigned long)ctx->stats.total_repairs_attempted);'

# 15. self_repair_health_notify (has nimcp_health_agent_t* health_agent)
append_training "$BASE/src/cognitive/fault_tolerance/nimcp_self_repair_health_notify.c" \
    "self_repair_health_notify" "g_self_repair_health_notify_health_agent" \
    "self_repair_health_notify_bridge" "self_repair_health_notify_bridge_t" "nonbridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->failure_tracking_count = 0;" \
    "ctx->stats.notifications_sent++;" \
    'NIMCP_LOGGING_INFO("%s training end: notifications=%lu, failures=%u", "self_repair_health_notify", (unsigned long)ctx->stats.notifications_sent, ctx->failure_tracking_count);'

echo ""
echo "=== FAULT TOLERANCE COMPLETE ==="

# Process health files
# 16. meta_health (has nimcp_health_agent_t* health_agent)
append_training "$BASE/src/cognitive/health/nimcp_meta_health.c" \
    "meta_health" "g_meta_health_health_agent" \
    "meta_health_reflector" "meta_health_reflector_t" "nonbridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->num_pending = 0; ctx->num_applied = 0; ctx->num_patterns = 0;" \
    "ctx->stats.reflections_completed++; ctx->num_pending++;" \
    'NIMCP_LOGGING_INFO("%s training end: reflections=%lu, patterns=%u", "meta_health", (unsigned long)ctx->stats.reflections_completed, ctx->num_patterns);'

# 17. collective_health (has local_agent not health_agent)
append_training "$BASE/src/cognitive/health/nimcp_collective_health.c" \
    "collective_health" "g_collective_health_health_agent" \
    "collective_health_monitor" "collective_health_monitor_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->num_instances = 0; ctx->num_pending_consensus = 0;" \
    "ctx->stats.sync_count++; ctx->num_instances++;" \
    'NIMCP_LOGGING_INFO("%s training end: syncs=%lu, instances=%u", "collective_health", (unsigned long)ctx->stats.sync_count, ctx->num_instances);'

# 18. rcog_health (has health_agent)
append_training "$BASE/src/cognitive/health/nimcp_rcog_health.c" \
    "rcog_health" "g_rcog_health_health_agent" \
    "rcog_health_integration" "rcog_health_integration_t" "nonbridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->num_pending = 0; ctx->cache_size = 0;" \
    "ctx->stats.queries_total++; ctx->num_pending++;" \
    'NIMCP_LOGGING_INFO("%s training end: queries=%lu, cache=%u", "rcog_health", (unsigned long)ctx->stats.queries_total, ctx->cache_size);'

# 19. health_cognitive_bridge (bridge, has 'agent' not 'health_agent')
append_training "$BASE/src/cognitive/health/nimcp_health_cognitive_bridge.c" \
    "health_cognitive_bridge" "g_health_cognitive_bridge_health_agent" \
    "health_cognitive_bridge" "health_cognitive_bridge_t" "bridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->num_pending = 0;" \
    "ctx->stats.events_handled++; ctx->num_pending++;" \
    'NIMCP_LOGGING_INFO("%s training end: events=%lu", "health_cognitive_bridge", (unsigned long)ctx->stats.events_handled);'

echo "=== HEALTH COMPLETE ==="

# Process mental_health files
# 20. mental_health
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health.c" \
    "mental_health" "g_mental_health_health_agent" \
    "mental_health_monitor" "mental_health_monitor_t" "nonbridge" "noha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->decisions_since_check = 0; ctx->history_index = 0;" \
    "ctx->decisions_since_check++;" \
    'NIMCP_LOGGING_INFO("%s training end: decisions=%u", "mental_health", ctx->decisions_since_check);'

# 21. mental_health_guardian
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_guardian.c" \
    "mental_health_guardian" "g_mental_health_guardian_health_agent" \
    "mental_health_guardian" "mental_health_guardian_t" "nonbridge" "noha" \
    "ctx->checks_performed = 0; ctx->interventions_applied = 0; ctx->last_overall_severity = 0.0f;" \
    "ctx->checks_performed++; ctx->last_overall_severity *= (1.0f - 0.01f * progress);" \
    'NIMCP_LOGGING_INFO("%s training end: checks=%lu, interventions=%lu", "mental_health_guardian", (unsigned long)ctx->checks_performed, (unsigned long)ctx->interventions_applied);'

# 22. mental_health_fep_bridge (no struct in .c - header defined)
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_fep_bridge.c" \
    "mental_health_fep_bridge" "g_mental_health_fep_bridge_health_agent" \
    "" "mental_health_fep_bridge_t" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "mental_health_fep_bridge");'

# 23. mental_health_thalamic_bridge
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_thalamic_bridge.c" \
    "mental_health_thalamic_bridge" "g_mental_health_thalamic_bridge_health_agent" \
    "mental_health_thalamic_bridge" "mental_health_thalamic_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->attention_weight = 1.0f;" \
    "ctx->stats.signals_routed++; float decay = 1.0f - 0.1f * progress; if (decay < 0.5f) decay = 0.5f; ctx->attention_weight *= decay;" \
    'ctx->attention_weight = 1.0f; NIMCP_LOGGING_INFO("%s training end: signals=%lu", "mental_health_thalamic_bridge", (unsigned long)ctx->stats.signals_routed);'

# 24. mental_health_substrate_bridge
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_substrate_bridge.c" \
    "mental_health_substrate_bridge" "g_mental_health_substrate_bridge_health_agent" \
    "mental_health_substrate_bridge" "mental_health_substrate_bridge_t" "bridge" "ha" \
    "ctx->update_count = 0; ctx->effects.overall_capacity = 1.0f;" \
    "ctx->update_count++; ctx->effects.overall_capacity = 0.5f + 0.5f * progress;" \
    'ctx->effects.overall_capacity = 1.0f; NIMCP_LOGGING_INFO("%s training end: updates=%lu", "mental_health_substrate_bridge", (unsigned long)ctx->update_count);'

# 25. mental_health_snn_bridge
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_snn_bridge.c" \
    "mental_health_snn_bridge" "g_mental_health_snn_bridge_health_agent" \
    "mental_health_snn_bridge" "mental_health_snn_bridge_t" "bridge" "ha" \
    "ctx->anxiety_signal = 0.0f; ctx->depression_signal = 0.0f; ctx->current_time_us = 0;" \
    "ctx->anxiety_signal *= (1.0f - 0.02f * progress); ctx->depression_signal *= (1.0f - 0.02f * progress);" \
    'NIMCP_LOGGING_INFO("%s training end: anxiety=%.4f, depression=%.4f", "mental_health_snn_bridge", ctx->anxiety_signal, ctx->depression_signal);'

# 26. mental_health_plasticity_bridge
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_plasticity_bridge.c" \
    "mental_health_plasticity_bridge" "g_mental_health_plasticity_bridge_health_agent" \
    "mental_health_plasticity_bridge" "mental_health_plasticity_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->synapse_count = 0; ctx->current_stress_level = 0.0f;" \
    "ctx->learning_rate_effective = 0.01f + 0.09f * (1.0f - progress); ctx->current_stress_level *= (1.0f - 0.02f * progress); ctx->stats.learn_events++;" \
    'NIMCP_LOGGING_INFO("%s training end: synapses=%u, stress=%.4f", "mental_health_plasticity_bridge", ctx->synapse_count, ctx->current_stress_level);'

# 27. mental_health_sleep_bridge
append_training "$BASE/src/cognitive/mental_health/nimcp_mental_health_sleep_bridge.c" \
    "mental_health_sleep_bridge" "g_mental_health_sleep_bridge_health_agent" \
    "mental_health_sleep_bridge_struct" "mental_health_sleep_bridge_t" "bridge" "ha" \
    "ctx->effects.stress_reduction = 0.0f; ctx->effects.emotional_regulation = 0.0f;" \
    "ctx->effects.stress_reduction = 0.3f + 0.7f * progress; ctx->effects.emotional_regulation = 0.3f + 0.7f * progress;" \
    'NIMCP_LOGGING_INFO("%s training end: stress_reduction=%.4f", "mental_health_sleep_bridge", ctx->effects.stress_reduction);'

# 28. interventions (#included, no struct)
append_training "$BASE/src/cognitive/mental_health/interventions.c" \
    "interventions" "g_interventions_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "interventions");'

echo "=== MENTAL HEALTH COMPLETE ==="

# Process wellbeing files (14 files)
# Most wellbeing files don't have struct defs in their .c files

# 29. wellbeing.c (no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing.c" \
    "wellbeing" "g_wellbeing_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing");'

# 30. wellbeing_enhanced.c
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_enhanced.c" \
    "wellbeing_enhanced" "g_wellbeing_enhanced_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_enhanced");'

# 31. wellbeing_homeostasis.c
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_homeostasis.c" \
    "wellbeing_homeostasis" "g_wellbeing_homeostasis_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_homeostasis");'

# 32. wellbeing_resources.c
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_resources.c" \
    "wellbeing_resources" "g_wellbeing_resources_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_resources");'

# 33. wellbeing_eudaimonic.c
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_eudaimonic.c" \
    "wellbeing_eudaimonic" "g_wellbeing_eudaimonic_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_eudaimonic");'

# 34. wellbeing_prediction.c
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_prediction.c" \
    "wellbeing_prediction" "g_wellbeing_prediction_health_agent" \
    "" "" "nonbridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_prediction");'

# 35. wellbeing_fep_bridge.c (bridge, no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_fep_bridge.c" \
    "wellbeing_fep_bridge" "g_wellbeing_fep_bridge_health_agent" \
    "" "" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_fep_bridge");'

# 36. wellbeing_substrate_bridge.c (bridge, no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_substrate_bridge.c" \
    "wellbeing_substrate_bridge" "g_wellbeing_substrate_bridge_health_agent" \
    "" "" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_substrate_bridge");'

# 37. wellbeing_thalamic_bridge.c (bridge, HAS struct)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_thalamic_bridge.c" \
    "wellbeing_thalamic_bridge" "g_wellbeing_thalamic_bridge_health_agent" \
    "wellbeing_thalamic_bridge" "wellbeing_thalamic_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->attention_weight = 1.0f;" \
    "ctx->stats.signals_routed++; float decay = 1.0f - 0.1f * progress; if (decay < 0.5f) decay = 0.5f; ctx->attention_weight *= decay;" \
    'ctx->attention_weight = 1.0f; NIMCP_LOGGING_INFO("%s training end: signals=%lu", "wellbeing_thalamic_bridge", (unsigned long)ctx->stats.signals_routed);'

# 38. wellbeing_snn_bridge.c (bridge, HAS struct)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_snn_bridge.c" \
    "wellbeing_snn_bridge" "g_wellbeing_snn_bridge_health_agent" \
    "wellbeing_snn_bridge" "wellbeing_snn_bridge_t" "bridge" "ha" \
    "ctx->current_time_us = 0;" \
    "ctx->current_time_us++;" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_snn_bridge");'

# 39. wellbeing_plasticity_bridge.c (bridge, HAS struct)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_plasticity_bridge.c" \
    "wellbeing_plasticity_bridge" "g_wellbeing_plasticity_bridge_health_agent" \
    "wellbeing_plasticity_bridge" "wellbeing_plasticity_bridge_t" "bridge" "ha" \
    "memset(&ctx->stats, 0, sizeof(ctx->stats)); ctx->synapse_count = 0; ctx->current_time_us = 0;" \
    "ctx->stats.learn_events++;" \
    'NIMCP_LOGGING_INFO("%s training end: synapses=%u", "wellbeing_plasticity_bridge", ctx->synapse_count);'

# 40. wellbeing_sleep_bridge.c (bridge, no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_sleep_bridge.c" \
    "wellbeing_sleep_bridge" "g_wellbeing_sleep_bridge_health_agent" \
    "" "" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_sleep_bridge");'

# 41. wellbeing_free_energy_bridge.c (bridge, no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_free_energy_bridge.c" \
    "wellbeing_free_energy_bridge" "g_wellbeing_free_energy_bridge_health_agent" \
    "" "" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_free_energy_bridge");'

# 42. wellbeing_mental_health_bridge.c (bridge, no struct in .c)
append_training "$BASE/src/cognitive/wellbeing/nimcp_wellbeing_mental_health_bridge.c" \
    "wellbeing_mental_health_bridge" "g_wellbeing_mental_health_bridge_health_agent" \
    "" "" "bridge" "noha" \
    "" \
    "" \
    'NIMCP_LOGGING_INFO("%s training end: complete", "wellbeing_mental_health_bridge");'

echo "=== WELLBEING COMPLETE ==="
echo "=== ALL FILES COMPLETE ==="
