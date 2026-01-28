#!/usr/bin/env python3
"""Phase 8: Add instance-level enhancements to ALL remaining files."""
import os, sys, re

BASE = "/home/bbrelin/nimcp"

def find_hb_end(content, pattern):
    """Find end of heartbeat function starting with pattern."""
    idx = content.find(pattern)
    if idx == -1:
        return -1
    bc = 0
    found = False
    for i in range(idx, len(content)):
        if content[i] == '{': bc += 1; found = True
        elif content[i] == '}':
            bc -= 1
            if found and bc == 0:
                return i + 1
    return -1

def add_health_to_struct(filepath, patterns):
    """Add health_agent field to struct in file."""
    with open(filepath) as f:
        content = f.read()
    if "nimcp_health_agent_t* health_agent;" in content:
        return
    for pat in patterns:
        idx = content.find(pat)
        if idx == -1:
            continue
        bc = 0; found = False
        for i in range(idx, len(content)):
            if content[i] == '{': bc += 1; found = True
            elif content[i] == '}':
                bc -= 1
                if found and bc == 0:
                    field = "\n    /* Phase 8: Instance-level health agent */\n    nimcp_health_agent_t* health_agent;\n"
                    content = content[:i] + field + content[i:]
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"  STRUCT+ {filepath}")
                    return
    print(f"  WARN: struct not found in {filepath} for patterns {patterns}")

def process(file_rel, hb_prefix, g_var, is_bridge, struct_type, short_p, log_mod,
            param_type=None, struct_file=None, struct_patterns=None,
            special_hb=None, bridge_fields=None):
    """Process a single file."""
    filepath = os.path.join(BASE, file_rel)
    if not os.path.exists(filepath):
        print(f"  MISS {file_rel}")
        return

    with open(filepath) as f:
        content = f.read()

    if f"{hb_prefix}_heartbeat_instance" in content:
        print(f"  SKIP {file_rel} (already done)")
        return

    # 1. Find heartbeat function end
    hb_pat = special_hb or f"static inline void {hb_prefix}_heartbeat("
    end = find_hb_end(content, hb_pat)
    if end == -1:
        print(f"  ERR: heartbeat not found in {file_rel}: {hb_pat}")
        return

    # 2. Insert heartbeat_instance after heartbeat
    hb_inst = f"""
static inline void {hb_prefix}_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{{
    if ({g_var}) {{
        nimcp_health_agent_heartbeat_ex({g_var}, operation, progress);
    }}
    if (instance_agent && instance_agent != {g_var}) {{
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }}
}}
"""
    content = content[:end] + "\n" + hb_inst + content[end:]

    # 3. Generate and append training code
    if is_bridge:
        pt = param_type or struct_type
        bf = bridge_fields or {}
        stats_f = bf.get("stats", "")
        state_f = bf.get("state", "")
        config_f = bf.get("config", "")
        counter_f = bf.get("counter", "")

        training = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Full Training
 * ============================================================================ */

/**
 * @brief Set instance-level health agent on bridge struct
 */
void {hb_prefix}_set_instance_health_agent(
    {pt}* bridge, nimcp_health_agent_t* agent) {{
    if (bridge) {{
        bridge->health_agent = agent;
    }}
}}

/**
 * @brief Begin training - reset counters, set flags, log start
 */
int {hb_prefix}_training_begin({pt}* bridge) {{
    if (!bridge) return -1;
    {hb_prefix}_heartbeat_instance(bridge->health_agent,
        "{short_p}_training_begin", 0.0f);
"""
        if counter_f: training += f"    bridge->{counter_f} = 0;\n"
        if stats_f: training += f"    bridge->{stats_f} = 0.0f;\n"
        if state_f: training += f"    bridge->{state_f} = 0.5f; /* Reset to neutral baseline */\n"
        training += f"""    NIMCP_LOGGING_INFO("[{log_mod}] Training begin: counters reset, baseline state initialized");
    return 0;
}}

/**
 * @brief Training step - clamp progress [0,1], adapt thresholds/weights, increment counters
 */
int {hb_prefix}_training_step({pt}* bridge, float progress) {{
    if (!bridge) return -1;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {hb_prefix}_heartbeat_instance(bridge->health_agent,
        "{short_p}_training_step", progress);
"""
        if config_f:
            training += f"""    /* Adapt threshold/weight based on training progress */
    float lr = bridge->{config_f};
    float adaptation = lr * (1.0f - progress) * 0.1f;
    bridge->{config_f} = lr + adaptation;
    if (bridge->{config_f} > 1.0f) bridge->{config_f} = 1.0f;
    if (bridge->{config_f} < 0.001f) bridge->{config_f} = 0.001f;
"""
        if state_f:
            training += f"""    /* Blend state toward training target */
    bridge->{state_f} = bridge->{state_f} * (1.0f - 0.01f) + progress * 0.01f;
"""
        if counter_f: training += f"    bridge->{counter_f}++;\n"
        training += f"""    return 0;
}}

/**
 * @brief End training - compute averages, clear flags, log metrics
 */
int {hb_prefix}_training_end({pt}* bridge) {{
    if (!bridge) return -1;
    {hb_prefix}_heartbeat_instance(bridge->health_agent,
        "{short_p}_training_end", 1.0f);
"""
        if state_f:
            training += f"""    /* Finalize state */
    if (bridge->{state_f} < 0.0f) bridge->{state_f} = 0.0f;
    if (bridge->{state_f} > 1.0f) bridge->{state_f} = 1.0f;
"""
        training += f"""    NIMCP_LOGGING_INFO("[{log_mod}] Training end: metrics finalized");
    return 0;
}}
"""
    else:
        # Non-bridge
        training = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Full Training
 * ============================================================================ */

/** Global instance health agent for {hb_prefix} (non-bridge fallback) */
static nimcp_health_agent_t* g_{hb_prefix}_instance_health_agent = NULL;

/**
 * @brief Set instance-level health agent (global fallback for non-bridge module)
 */
void {hb_prefix}_set_instance_health_agent(void* ctx, nimcp_health_agent_t* agent) {{
    (void)ctx;
    g_{hb_prefix}_instance_health_agent = agent;
}}

/**
 * @brief Begin training - reset module state, log start
 */
int {hb_prefix}_training_begin(void* ctx) {{
    if (!ctx) return -1;
    {hb_prefix}_heartbeat_instance(g_{hb_prefix}_instance_health_agent,
        "{short_p}_training_begin", 0.0f);
    NIMCP_LOGGING_INFO("[{log_mod}] Training begin: module state reset");
    return 0;
}}

/**
 * @brief Training step - clamp progress [0,1], adapt internal parameters
 */
int {hb_prefix}_training_step(void* ctx, float progress) {{
    if (!ctx) return -1;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {hb_prefix}_heartbeat_instance(g_{hb_prefix}_instance_health_agent,
        "{short_p}_training_step", progress);
    return 0;
}}

/**
 * @brief End training - compute final metrics, log results
 */
int {hb_prefix}_training_end(void* ctx) {{
    if (!ctx) return -1;
    {hb_prefix}_heartbeat_instance(g_{hb_prefix}_instance_health_agent,
        "{short_p}_training_end", 1.0f);
    NIMCP_LOGGING_INFO("[{log_mod}] Training end: metrics finalized");
    return 0;
}}
"""

    content = content.rstrip() + "\n" + training
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  OK {file_rel}")

    # Add health_agent to struct if bridge
    if is_bridge and struct_file and struct_patterns:
        sf = os.path.join(BASE, struct_file)
        add_health_to_struct(sf, struct_patterns)

# =================== PROCESS ALL FILES ===================

# Already done: introspection.c, introspection_fep_bridge.c

# 3. introspection_substrate_bridge - BRIDGE (struct in header)
process("src/cognitive/introspection/nimcp_introspection_substrate_bridge.c",
    "introspection_substrate_bridge", "g_introspection_substrate_bridge_health_agent",
    True, "introspection_substrate_bridge_t", "intro_sub", "INTRO_SUBSTRATE",
    struct_file="include/cognitive/introspection/nimcp_introspection_substrate_bridge.h",
    struct_patterns=["struct introspection_substrate_bridge {"],
    bridge_fields={"stats": "stats.avg_atp", "state": "effects.self_awareness_depth",
                   "config": "config.atp_sensitivity", "counter": "stats.update_count"})

# 4. introspection_thalamic_bridge - BRIDGE (struct in .c)
process("src/cognitive/introspection/nimcp_introspection_thalamic_bridge.c",
    "introspection_thalamic_bridge", "g_introspection_thalamic_bridge_health_agent",
    True, "introspection_thalamic_bridge_t", "intro_thal", "INTRO_THALAMIC",
    struct_file="src/cognitive/introspection/nimcp_introspection_thalamic_bridge.c",
    struct_patterns=["struct introspection_thalamic_bridge {"],
    bridge_fields={"stats": "stats.avg_depth", "state": "attention_weight",
                   "config": "config.min_urgency_threshold", "counter": "stats.monitors_routed"})

# 5. introspection_plasticity_bridge - BRIDGE (struct in .c)
process("src/cognitive/introspection/nimcp_introspection_plasticity_bridge.c",
    "introspection_plasticity_bridge", "g_introspection_plasticity_bridge_health_agent",
    True, "introspection_plasticity_bridge_t", "intro_plast", "INTRO_PLASTICITY",
    struct_file="src/cognitive/introspection/nimcp_introspection_plasticity_bridge.c",
    struct_patterns=["struct introspection_plasticity_bridge {"],
    bridge_fields={"stats": "stats.mean_weight_change", "state": "calibration.confidence_calibration",
                   "config": "config.base_learning_rate", "counter": "stats.total_learning_events"})

# 6. introspection_sleep_bridge - BRIDGE (struct in .c, handle type)
process("src/cognitive/introspection/nimcp_introspection_sleep_bridge.c",
    "introspection_sleep_bridge", "g_introspection_sleep_bridge_health_agent",
    True, "introspection_sleep_bridge_t", "intro_sleep", "INTRO_SLEEP",
    struct_file="src/cognitive/introspection/nimcp_introspection_sleep_bridge.c",
    struct_patterns=["struct introspection_sleep_bridge_struct {"],
    bridge_fields={"state": "effects.metacognitive_accuracy_factor",
                   "config": "config.modulation_strength"})

# 7. introspection_snn_bridge - BRIDGE (struct in .c)
process("src/cognitive/introspection/nimcp_introspection_snn_bridge.c",
    "introspection_snn_bridge", "g_introspection_snn_bridge_health_agent",
    True, "introspection_snn_bridge_t", "intro_snn", "INTRO_SNN",
    struct_file="src/cognitive/introspection/nimcp_introspection_snn_bridge.c",
    struct_patterns=["struct introspection_snn_bridge {"],
    bridge_fields={"stats": "stats.mean_confidence", "state": "uncertainty_signal",
                   "config": "config.encoding_gain", "counter": "stats.total_evaluations"})

# 8. connectivity_health - NON-BRIDGE
process("src/cognitive/introspection/nimcp_connectivity_health.c",
    "connectivity_health", "g_connectivity_health_health_agent",
    False, None, "conn_health", "CONNECTIVITY_HEALTH")

# 9. consciousness_metrics - NON-BRIDGE
process("src/cognitive/introspection/nimcp_consciousness_metrics.c",
    "consciousness_metrics", "g_consciousness_metrics_health_agent",
    False, None, "consc_met", "CONSCIOUSNESS_METRICS")

# 10. ensemble_uncertainty - NON-BRIDGE
process("src/cognitive/introspection/nimcp_ensemble_uncertainty.c",
    "ensemble_uncertainty", "g_ensemble_uncertainty_health_agent",
    False, None, "ens_uncert", "ENSEMBLE_UNCERTAINTY")

# 11. ensemble_uncertainty_pink_noise_bridge - BRIDGE (struct in header)
process("src/cognitive/introspection/nimcp_ensemble_uncertainty_pink_noise_bridge.c",
    "ensemble_uncertainty_pink_noise_bridge",
    "g_ensemble_uncertainty_pink_noise_bridge_health_agent",
    True, "ensemble_pink_bridge_t", "ens_pink", "ENS_PINK",
    struct_file="include/cognitive/introspection/nimcp_ensemble_uncertainty_pink_noise_bridge.h",
    struct_patterns=["} ensemble_pink_bridge_t;"],
    bridge_fields={"stats": "stats.avg_amplitude", "state": "state.current_amplitude",
                   "config": "config.base_amplitude", "counter": "stats.total_injections"})

# 12. temporal_patterns - NON-BRIDGE
process("src/cognitive/introspection/nimcp_temporal_patterns.c",
    "temporal_patterns", "g_temporal_patterns_health_agent",
    False, None, "temp_pat", "TEMPORAL_PATTERNS")

# =========================================================================
# KNOWLEDGE (11 files)
# =========================================================================

# 13. knowledge - NON-BRIDGE
process("src/cognitive/knowledge/nimcp_knowledge.c",
    "knowledge", "g_knowledge_health_agent",
    False, None, "knowledge", "KNOWLEDGE")

# 14. kg_reader - NON-BRIDGE
process("src/cognitive/knowledge/nimcp_kg_reader.c",
    "kg_reader", "g_kg_reader_health_agent",
    False, None, "kg_reader", "KG_READER")

# 15. knowledge_cow - NON-BRIDGE
process("src/cognitive/knowledge/nimcp_knowledge_cow.c",
    "knowledge_cow", "g_knowledge_cow_health_agent",
    False, None, "know_cow", "KNOWLEDGE_COW")

# 16. knowledge_fractal - NON-BRIDGE
process("src/cognitive/knowledge/nimcp_knowledge_fractal.c",
    "knowledge_fractal", "g_knowledge_fractal_health_agent",
    False, None, "know_frac", "KNOWLEDGE_FRACTAL")

# 17. knowledge_hyperbolic - NON-BRIDGE
process("src/cognitive/knowledge/nimcp_knowledge_hyperbolic.c",
    "knowledge_hyperbolic", "g_knowledge_hyperbolic_health_agent",
    False, None, "know_hyper", "KNOWLEDGE_HYPERBOLIC")

# 18. knowledge_fep_bridge - BRIDGE (struct in header)
process("src/cognitive/knowledge/nimcp_knowledge_fep_bridge.c",
    "knowledge_fep_bridge", "g_knowledge_fep_bridge_health_agent",
    True, "knowledge_fep_bridge_t", "know_fep", "KNOWLEDGE_FEP",
    struct_file="include/cognitive/knowledge/nimcp_knowledge_fep_bridge.h",
    struct_patterns=["struct knowledge_fep_bridge {"],
    bridge_fields={"stats": "stats.avg_precision", "state": "state.current_precision",
                   "config": "config.meta_learning_rate", "counter": "stats.precision_updates_total"})

# 19. knowledge_substrate_bridge - BRIDGE (struct in .c)
process("src/cognitive/knowledge/nimcp_knowledge_substrate_bridge.c",
    "knowledge_substrate_bridge", "g_knowledge_substrate_bridge_health_agent",
    True, "knowledge_substrate_bridge_t", "know_sub", "KNOWLEDGE_SUBSTRATE",
    struct_file="src/cognitive/knowledge/nimcp_knowledge_substrate_bridge.c",
    struct_patterns=["struct knowledge_substrate_bridge {"],
    bridge_fields={"state": "effects.knowledge_retrieval_speed",
                   "config": "config.atp_sensitivity", "counter": "update_count"})

# 20. knowledge_thalamic_bridge - BRIDGE (struct in .c)
process("src/cognitive/knowledge/nimcp_knowledge_thalamic_bridge.c",
    "knowledge_thalamic_bridge", "g_knowledge_thalamic_bridge_health_agent",
    True, "knowledge_thalamic_bridge_t", "know_thal", "KNOWLEDGE_THALAMIC",
    struct_file="src/cognitive/knowledge/nimcp_knowledge_thalamic_bridge.c",
    struct_patterns=["struct knowledge_thalamic_bridge {"],
    bridge_fields={"stats": "stats.avg_relevance", "state": "attention_weight",
                   "config": "config.min_relevance_threshold", "counter": "stats.queries_routed"})

# 21. knowledge_plasticity_bridge - BRIDGE (struct in .c)
process("src/cognitive/knowledge/nimcp_knowledge_plasticity_bridge.c",
    "knowledge_plasticity_bridge", "g_knowledge_plasticity_bridge_health_agent",
    True, "knowledge_plasticity_bridge_t", "know_plast", "KNOWLEDGE_PLASTICITY",
    struct_file="src/cognitive/knowledge/nimcp_knowledge_plasticity_bridge.c",
    struct_patterns=["struct knowledge_plasticity_bridge {"],
    bridge_fields={"stats": "stats.mean_weight_change", "state": "consolidation_state.consolidation_progress",
                   "config": "config.base_learning_rate", "counter": "stats.total_learning_events"})

# 22. knowledge_sleep_bridge - BRIDGE (struct in .c, handle)
process("src/cognitive/knowledge/nimcp_knowledge_sleep_bridge.c",
    "knowledge_sleep_bridge", "g_knowledge_sleep_bridge_health_agent",
    True, "knowledge_sleep_bridge_t", "know_sleep", "KNOWLEDGE_SLEEP",
    struct_file="src/cognitive/knowledge/nimcp_knowledge_sleep_bridge.c",
    struct_patterns=["struct knowledge_sleep_bridge_struct {"],
    bridge_fields={"state": "effects.consolidation_factor",
                   "config": "config.modulation_strength"})

# 23. knowledge_snn_bridge - BRIDGE (struct in .c)
process("src/cognitive/knowledge/nimcp_knowledge_snn_bridge.c",
    "knowledge_snn_bridge", "g_knowledge_snn_bridge_health_agent",
    True, "knowledge_snn_bridge_t", "know_snn", "KNOWLEDGE_SNN",
    struct_file="src/cognitive/knowledge/nimcp_knowledge_snn_bridge.c",
    struct_patterns=["struct knowledge_snn_bridge {"],
    bridge_fields={"stats": "stats.mean_confidence", "state": "last_insight.confidence",
                   "config": "config.encoding_gain", "counter": "stats.total_evaluations"})

# =========================================================================
# INNER DIALOGUE (4 files)
# =========================================================================

# 24. inner_dialogue - NON-BRIDGE (has engine struct but special heartbeat)
process("src/cognitive/inner_dialogue/nimcp_inner_dialogue.c",
    "engine", "g_inner_dialogue_health_agent",
    False, None, "inner_dlg", "INNER_DIALOGUE")

# 25. convergence - NON-BRIDGE
process("src/cognitive/inner_dialogue/nimcp_inner_dialogue_convergence.c",
    "convergence", "g_convergence_health_agent",
    False, None, "convergence", "CONVERGENCE")

# 26. perspective - NON-BRIDGE
process("src/cognitive/inner_dialogue/nimcp_inner_dialogue_perspective.c",
    "perspective", "g_perspective_health_agent",
    False, None, "perspective", "PERSPECTIVE")

# 27. turn - NON-BRIDGE
process("src/cognitive/inner_dialogue/nimcp_inner_dialogue_turn.c",
    "turn", "g_turn_health_agent",
    False, None, "turn", "TURN")

print("\nDone!")
