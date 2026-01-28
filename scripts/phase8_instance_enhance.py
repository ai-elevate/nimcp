#!/usr/bin/env python3
"""Phase 8: Add instance-level health agent enhancements to all 27 files."""

import os
import re

BASE = "/home/bbrelin/nimcp"

# Each entry: (file_path, heartbeat_prefix, global_var, is_bridge, struct_type, short_prefix, struct_fields_for_training)
# For bridges: struct_type is the typed bridge struct
# For non-bridges: struct_type is used for the void* cast, short_prefix for heartbeat label
FILES = [
    # =========================================================================
    # INTROSPECTION (12 files)
    # =========================================================================
    # 1. nimcp_introspection.c - NON-BRIDGE (opaque handle)
    {
        "file": "src/cognitive/introspection/nimcp_introspection.c",
        "hb_prefix": "introspection",
        "g_var": "g_introspection_health_agent",
        "is_bridge": False,
        "struct_type": None,  # opaque - uses void*
        "short": "introspection",
        "log_mod": "INTROSPECTION",
        "training_fields": {
            "stats": "stats.queries_total",
            "avg": "stats.avg_query_time_ms",
            "counter": "stats.queries_active_population",
        },
    },
    # 2. nimcp_introspection_fep_bridge.c - BRIDGE
    {
        "file": "src/cognitive/introspection/nimcp_introspection_fep_bridge.c",
        "hb_prefix": "introspection_fep_bridge",
        "g_var": "g_introspection_fep_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "introspection_fep_bridge_t",
        "short": "intro_fep",
        "log_mod": "INTROSPECTION_FEP",
        "training_fields": {
            "stats_field": "stats.avg_precision",
            "state_field": "state.current_precision",
            "config_field": "config.meta_learning_rate",
            "counter": "stats.precision_estimates_total",
        },
    },
    # 3. nimcp_introspection_substrate_bridge.c - BRIDGE (non-opaque struct)
    {
        "file": "src/cognitive/introspection/nimcp_introspection_substrate_bridge.c",
        "hb_prefix": "introspection_substrate_bridge",
        "g_var": "g_introspection_substrate_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "introspection_substrate_bridge_t",
        "short": "intro_sub",
        "log_mod": "INTRO_SUBSTRATE",
        "training_fields": {
            "stats_field": "stats.avg_atp",
            "effects_field": "effects.self_awareness_depth",
            "config_field": "config.atp_sensitivity",
            "counter": "stats.update_count",
        },
    },
    # 4. nimcp_introspection_thalamic_bridge.c - BRIDGE (opaque struct in .c)
    {
        "file": "src/cognitive/introspection/nimcp_introspection_thalamic_bridge.c",
        "hb_prefix": "introspection_thalamic_bridge",
        "g_var": "g_introspection_thalamic_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "introspection_thalamic_bridge_t",
        "short": "intro_thal",
        "log_mod": "INTRO_THALAMIC",
        "training_fields": {
            "stats_field": "stats.avg_depth",
            "state_field": "attention_weight",
            "config_field": "config.min_urgency_threshold",
            "counter": "stats.monitors_routed",
        },
    },
    # 5. nimcp_introspection_plasticity_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/introspection/nimcp_introspection_plasticity_bridge.c",
        "hb_prefix": "introspection_plasticity_bridge",
        "g_var": "g_introspection_plasticity_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "introspection_plasticity_bridge_t",
        "short": "intro_plast",
        "log_mod": "INTRO_PLASTICITY",
        "training_fields": {
            "stats_field": "stats.mean_weight_change",
            "state_field": "calibration.confidence_calibration",
            "config_field": "config.base_learning_rate",
            "counter": "stats.total_learning_events",
        },
    },
    # 6. nimcp_introspection_sleep_bridge.c - BRIDGE (opaque handle)
    {
        "file": "src/cognitive/introspection/nimcp_introspection_sleep_bridge.c",
        "hb_prefix": "introspection_sleep_bridge",
        "g_var": "g_introspection_sleep_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "struct introspection_sleep_bridge_struct",
        "short": "intro_sleep",
        "log_mod": "INTRO_SLEEP",
        "training_fields": {
            "effects_field": "effects.metacognitive_accuracy_factor",
            "config_field": "config.modulation_strength",
            "state_field": "effects.consciousness_level_factor",
        },
        "param_type": "introspection_sleep_bridge_t",  # handle type
    },
    # 7. nimcp_introspection_snn_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/introspection/nimcp_introspection_snn_bridge.c",
        "hb_prefix": "introspection_snn_bridge",
        "g_var": "g_introspection_snn_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "introspection_snn_bridge_t",
        "short": "intro_snn",
        "log_mod": "INTRO_SNN",
        "training_fields": {
            "stats_field": "stats.mean_confidence",
            "state_field": "uncertainty_signal",
            "config_field": "config.encoding_gain",
            "counter": "stats.total_evaluations",
        },
    },
    # 8. nimcp_connectivity_health.c - NON-BRIDGE
    {
        "file": "src/cognitive/introspection/nimcp_connectivity_health.c",
        "hb_prefix": "connectivity_health",
        "g_var": "g_connectivity_health_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "conn_health",
        "log_mod": "CONNECTIVITY_HEALTH",
        "training_fields": {},
    },
    # 9. nimcp_consciousness_metrics.c - NON-BRIDGE
    {
        "file": "src/cognitive/introspection/nimcp_consciousness_metrics.c",
        "hb_prefix": "consciousness_metrics",
        "g_var": "g_consciousness_metrics_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "consc_met",
        "log_mod": "CONSCIOUSNESS_METRICS",
        "training_fields": {},
    },
    # 10. nimcp_ensemble_uncertainty.c - NON-BRIDGE (opaque handle)
    {
        "file": "src/cognitive/introspection/nimcp_ensemble_uncertainty.c",
        "hb_prefix": "ensemble_uncertainty",
        "g_var": "g_ensemble_uncertainty_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "ens_uncert",
        "log_mod": "ENSEMBLE_UNCERTAINTY",
        "training_fields": {},
    },
    # 11. nimcp_ensemble_uncertainty_pink_noise_bridge.c - BRIDGE (non-opaque struct)
    {
        "file": "src/cognitive/introspection/nimcp_ensemble_uncertainty_pink_noise_bridge.c",
        "hb_prefix": "ensemble_uncertainty_pink_noise_bridge",
        "g_var": "g_ensemble_uncertainty_pink_noise_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "ensemble_pink_bridge_t",
        "short": "ens_pink",
        "log_mod": "ENS_PINK",
        "training_fields": {
            "stats_field": "stats.avg_amplitude",
            "state_field": "state.current_amplitude",
            "config_field": "config.base_amplitude",
            "counter": "stats.total_injections",
        },
    },
    # 12. nimcp_temporal_patterns.c - NON-BRIDGE
    {
        "file": "src/cognitive/introspection/nimcp_temporal_patterns.c",
        "hb_prefix": "temporal_patterns",
        "g_var": "g_temporal_patterns_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "temp_pat",
        "log_mod": "TEMPORAL_PATTERNS",
        "training_fields": {},
    },
    # =========================================================================
    # KNOWLEDGE (11 files)
    # =========================================================================
    # 13. nimcp_knowledge.c - NON-BRIDGE
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge.c",
        "hb_prefix": "knowledge",
        "g_var": "g_knowledge_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "knowledge",
        "log_mod": "KNOWLEDGE",
        "training_fields": {},
    },
    # 14. nimcp_kg_reader.c - NON-BRIDGE (opaque kg_reader_t)
    {
        "file": "src/cognitive/knowledge/nimcp_kg_reader.c",
        "hb_prefix": "kg_reader",
        "g_var": "g_kg_reader_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "kg_reader",
        "log_mod": "KG_READER",
        "training_fields": {},
    },
    # 15. nimcp_knowledge_cow.c - NON-BRIDGE
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_cow.c",
        "hb_prefix": "knowledge_cow",
        "g_var": "g_knowledge_cow_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "know_cow",
        "log_mod": "KNOWLEDGE_COW",
        "training_fields": {},
    },
    # 16. nimcp_knowledge_fractal.c - NON-BRIDGE
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_fractal.c",
        "hb_prefix": "knowledge_fractal",
        "g_var": "g_knowledge_fractal_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "know_frac",
        "log_mod": "KNOWLEDGE_FRACTAL",
        "training_fields": {},
    },
    # 17. nimcp_knowledge_hyperbolic.c - NON-BRIDGE
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_hyperbolic.c",
        "hb_prefix": "knowledge_hyperbolic",
        "g_var": "g_knowledge_hyperbolic_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "know_hyper",
        "log_mod": "KNOWLEDGE_HYPERBOLIC",
        "training_fields": {},
    },
    # 18. nimcp_knowledge_fep_bridge.c - BRIDGE
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_fep_bridge.c",
        "hb_prefix": "knowledge_fep_bridge",
        "g_var": "g_knowledge_fep_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "knowledge_fep_bridge_t",
        "short": "know_fep",
        "log_mod": "KNOWLEDGE_FEP",
        "training_fields": {
            "stats_field": "stats.avg_precision",
            "state_field": "state.current_precision",
            "config_field": "config.meta_learning_rate",
            "counter": "stats.precision_updates_total",
        },
    },
    # 19. nimcp_knowledge_substrate_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_substrate_bridge.c",
        "hb_prefix": "knowledge_substrate_bridge",
        "g_var": "g_knowledge_substrate_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "knowledge_substrate_bridge_t",
        "short": "know_sub",
        "log_mod": "KNOWLEDGE_SUBSTRATE",
        "training_fields": {
            "effects_field": "effects.knowledge_retrieval_speed",
            "config_field": "config.atp_sensitivity",
            "counter": "update_count",
        },
    },
    # 20. nimcp_knowledge_thalamic_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_thalamic_bridge.c",
        "hb_prefix": "knowledge_thalamic_bridge",
        "g_var": "g_knowledge_thalamic_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "knowledge_thalamic_bridge_t",
        "short": "know_thal",
        "log_mod": "KNOWLEDGE_THALAMIC",
        "training_fields": {
            "stats_field": "stats.avg_relevance",
            "state_field": "attention_weight",
            "config_field": "config.min_relevance_threshold",
            "counter": "stats.queries_routed",
        },
    },
    # 21. nimcp_knowledge_plasticity_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_plasticity_bridge.c",
        "hb_prefix": "knowledge_plasticity_bridge",
        "g_var": "g_knowledge_plasticity_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "knowledge_plasticity_bridge_t",
        "short": "know_plast",
        "log_mod": "KNOWLEDGE_PLASTICITY",
        "training_fields": {
            "stats_field": "stats.mean_weight_change",
            "state_field": "consolidation_state.consolidation_progress",
            "config_field": "config.base_learning_rate",
            "counter": "stats.total_learning_events",
        },
    },
    # 22. nimcp_knowledge_sleep_bridge.c - BRIDGE (opaque handle)
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_sleep_bridge.c",
        "hb_prefix": "knowledge_sleep_bridge",
        "g_var": "g_knowledge_sleep_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "struct knowledge_sleep_bridge_struct",
        "short": "know_sleep",
        "log_mod": "KNOWLEDGE_SLEEP",
        "training_fields": {
            "effects_field": "effects.consolidation_factor",
            "config_field": "config.modulation_strength",
            "state_field": "effects.retrieval_factor",
        },
        "param_type": "knowledge_sleep_bridge_t",
    },
    # 23. nimcp_knowledge_snn_bridge.c - BRIDGE (opaque)
    {
        "file": "src/cognitive/knowledge/nimcp_knowledge_snn_bridge.c",
        "hb_prefix": "knowledge_snn_bridge",
        "g_var": "g_knowledge_snn_bridge_health_agent",
        "is_bridge": True,
        "struct_type": "knowledge_snn_bridge_t",
        "short": "know_snn",
        "log_mod": "KNOWLEDGE_SNN",
        "training_fields": {
            "stats_field": "stats.mean_confidence",
            "state_field": "last_insight.confidence",
            "config_field": "config.encoding_gain",
            "counter": "stats.total_evaluations",
        },
    },
    # =========================================================================
    # INNER DIALOGUE (4 files)
    # =========================================================================
    # 24. nimcp_inner_dialogue.c - NON-BRIDGE (opaque engine)
    {
        "file": "src/cognitive/inner_dialogue/nimcp_inner_dialogue.c",
        "hb_prefix": "engine",  # this uses engine_heartbeat but that takes engine* param
        "g_var": "g_inner_dialogue_health_agent",
        "is_bridge": False,
        "struct_type": "inner_dialogue_engine_t",
        "short": "inner_dlg",
        "log_mod": "INNER_DIALOGUE",
        "training_fields": {
            "state_field": "turn_number",
            "config_field": "config.max_turns",
        },
        "special_heartbeat": True,  # engine_heartbeat has different signature
    },
    # 25. nimcp_inner_dialogue_convergence.c - NON-BRIDGE
    {
        "file": "src/cognitive/inner_dialogue/nimcp_inner_dialogue_convergence.c",
        "hb_prefix": "convergence",
        "g_var": "g_convergence_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "convergence",
        "log_mod": "CONVERGENCE",
        "training_fields": {},
    },
    # 26. nimcp_inner_dialogue_perspective.c - NON-BRIDGE
    {
        "file": "src/cognitive/inner_dialogue/nimcp_inner_dialogue_perspective.c",
        "hb_prefix": "perspective",
        "g_var": "g_perspective_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "perspective",
        "log_mod": "PERSPECTIVE",
        "training_fields": {},
    },
    # 27. nimcp_inner_dialogue_turn.c - NON-BRIDGE
    {
        "file": "src/cognitive/inner_dialogue/nimcp_inner_dialogue_turn.c",
        "hb_prefix": "turn",
        "g_var": "g_turn_health_agent",
        "is_bridge": False,
        "struct_type": None,
        "short": "turn",
        "log_mod": "TURN",
        "training_fields": {},
    },
]


def gen_heartbeat_instance(f):
    """Generate heartbeat_instance function."""
    prefix = f["hb_prefix"]
    g_var = f["g_var"]
    return f"""
static inline void {prefix}_heartbeat_instance(
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


def gen_bridge_instance_code(f):
    """Generate instance-level code for bridge files."""
    prefix = f["hb_prefix"]
    short = f["short"]
    struct_type = f["struct_type"]
    g_var = f["g_var"]
    param_type = f.get("param_type", struct_type)
    tf = f["training_fields"]

    # Use the first available field for training interactions
    stats_f = tf.get("stats_field", tf.get("effects_field", ""))
    state_f = tf.get("state_field", tf.get("effects_field", ""))
    config_f = tf.get("config_field", "")
    counter_f = tf.get("counter", "")

    code = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Full Training
 * ============================================================================ */

/**
 * @brief Set instance-level health agent on bridge struct
 */
void {prefix}_set_instance_health_agent({param_type}* bridge, nimcp_health_agent_t* agent) {{
    if (bridge) {{
        bridge->health_agent = agent;
    }}
}}

/**
 * @brief Begin training - reset counters, set flags, log start
 */
int {prefix}_training_begin({param_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{short}_training_begin", 0.0f);
"""
    # Reset counters & stats based on available fields
    if counter_f:
        code += f"    bridge->{counter_f} = 0;\n"
    if stats_f:
        code += f"    bridge->{stats_f} = 0.0f;\n"
    if state_f:
        code += f"    /* Mark training active via state */\n"
        code += f"    bridge->{state_f} = 0.5f; /* Reset to neutral baseline */\n"
    code += f"""    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training begin: counters reset, baseline state initialized");
    return 0;
}}

/**
 * @brief Training step - clamp progress [0,1], adapt thresholds/weights, increment counters
 */
int {prefix}_training_step({param_type}* bridge, float progress) {{
    if (!bridge) return -1;
    /* Clamp progress to [0,1] */
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance(bridge->health_agent, "{short}_training_step", progress);
"""
    if config_f:
        code += f"""    /* Adapt threshold/weight based on training progress */
    float lr = bridge->{config_f};
    float adaptation = lr * (1.0f - progress) * 0.1f;
    bridge->{config_f} = lr + adaptation;
    if (bridge->{config_f} > 1.0f) bridge->{config_f} = 1.0f;
    if (bridge->{config_f} < 0.001f) bridge->{config_f} = 0.001f;
"""
    if state_f:
        code += f"""    /* Blend state toward training target */
    bridge->{state_f} = bridge->{state_f} * (1.0f - 0.01f) + progress * 0.01f;
"""
    if counter_f:
        code += f"    bridge->{counter_f}++;\n"
    code += f"""    return 0;
}}

/**
 * @brief End training - compute averages, clear flags, log metrics
 */
int {prefix}_training_end({param_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{short}_training_end", 1.0f);
"""
    if stats_f and counter_f:
        code += f"""    /* Compute final averages */
    if (bridge->{counter_f} > 0) {{
        bridge->{stats_f} = bridge->{stats_f} / (float)(bridge->{counter_f} > 0 ? 1 : 1);
    }}
"""
    if state_f:
        code += f"""    /* Finalize state */
    if (bridge->{state_f} < 0.0f) bridge->{state_f} = 0.0f;
    if (bridge->{state_f} > 1.0f) bridge->{state_f} = 1.0f;
"""
    code += f"""    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training end: metrics finalized");
    return 0;
}}
"""
    return code


def gen_nonbridge_instance_code(f):
    """Generate instance-level code for non-bridge files."""
    prefix = f["hb_prefix"]
    short = f["short"]
    g_var = f["g_var"]
    struct_type = f.get("struct_type")

    # For inner_dialogue.c which has a concrete opaque struct
    if struct_type and struct_type != "None":
        return gen_nonbridge_with_struct(f)

    # For truly opaque/no-struct non-bridge files - use void* with global fallback
    code = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Full Training
 * ============================================================================ */

/** Global instance health agent for {prefix} (non-bridge fallback) */
static nimcp_health_agent_t* g_{prefix}_instance_health_agent = NULL;

/**
 * @brief Set instance-level health agent (global fallback for non-bridge module)
 */
void {prefix}_set_instance_health_agent(void* ctx, nimcp_health_agent_t* agent) {{
    (void)ctx;
    g_{prefix}_instance_health_agent = agent;
}}

/**
 * @brief Begin training - reset module state, log start
 */
int {prefix}_training_begin(void* ctx) {{
    if (!ctx) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_begin", 0.0f);
    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training begin: module state reset");
    return 0;
}}

/**
 * @brief Training step - clamp progress [0,1], adapt internal parameters
 */
int {prefix}_training_step(void* ctx, float progress) {{
    if (!ctx) return -1;
    /* Clamp progress to [0,1] */
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_step", progress);
    return 0;
}}

/**
 * @brief End training - compute final metrics, log results
 */
int {prefix}_training_end(void* ctx) {{
    if (!ctx) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_end", 1.0f);
    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training end: metrics finalized");
    return 0;
}}
"""
    return code


def gen_nonbridge_with_struct(f):
    """Generate instance-level code for non-bridge files with known struct."""
    prefix = f["hb_prefix"]
    short = f["short"]
    g_var = f["g_var"]
    struct_type = f["struct_type"]
    tf = f.get("training_fields", {})

    state_f = tf.get("state_field", "")
    config_f = tf.get("config_field", "")

    # inner_dialogue has a special heartbeat signature
    special = f.get("special_heartbeat", False)

    code = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Full Training
 * ============================================================================ */

/** Global instance health agent for {prefix} */
static nimcp_health_agent_t* g_{prefix}_instance_health_agent = NULL;

/**
 * @brief Set instance-level health agent for {prefix} module
 */
void {prefix}_set_instance_health_agent(void* ctx, nimcp_health_agent_t* agent) {{
    (void)ctx;
    g_{prefix}_instance_health_agent = agent;
}}

/**
 * @brief Begin training - reset counters, set flags, log start
 */
int {prefix}_training_begin(void* ctx) {{
    if (!ctx) return -1;
    {struct_type}* mod = ({struct_type}*)ctx;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_begin", 0.0f);
"""
    if state_f:
        code += f"    mod->{state_f} = 0; /* Reset to baseline */\n"
    code += f"""    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training begin: state reset for module instance");
    (void)mod;
    return 0;
}}

/**
 * @brief Training step - clamp progress [0,1], adapt parameters
 */
int {prefix}_training_step(void* ctx, float progress) {{
    if (!ctx) return -1;
    {struct_type}* mod = ({struct_type}*)ctx;
    /* Clamp progress to [0,1] */
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_step", progress);
"""
    if config_f:
        code += f"""    /* Adapt configuration based on training progress */
    float base = (float)mod->{config_f};
    float adapted = base + (1.0f - progress) * 0.01f * base;
    if (adapted > 0.0f) mod->{config_f} = (uint32_t)(adapted > 0 ? adapted : base);
"""
    if state_f:
        code += f"    mod->{state_f} = (uint32_t)(progress * 100.0f);\n"
    code += f"""    (void)mod;
    return 0;
}}

/**
 * @brief End training - compute averages, clear flags, log metrics
 */
int {prefix}_training_end(void* ctx) {{
    if (!ctx) return -1;
    {struct_type}* mod = ({struct_type}*)ctx;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short}_training_end", 1.0f);
    NIMCP_LOGGING_INFO("[{f['log_mod']}] Training end: module metrics finalized");
    (void)mod;
    return 0;
}}
"""
    return code


def process_file(f):
    filepath = os.path.join(BASE, f["file"])
    with open(filepath, "r") as fh:
        content = fh.read()

    # 1. Check if heartbeat_instance already exists
    prefix = f["hb_prefix"]
    if f"{prefix}_heartbeat_instance" in content:
        print(f"  SKIP {f['file']} (heartbeat_instance already exists)")
        return False

    # 2. Find the existing heartbeat function to insert heartbeat_instance after it
    # Pattern: static inline void PREFIX_heartbeat(const char* operation, float progress) { ... }
    hb_pattern = f"static inline void {prefix}_heartbeat("

    # For inner_dialogue engine_heartbeat has different signature
    if f.get("special_heartbeat"):
        hb_pattern = "static inline void engine_heartbeat("

    idx = content.find(hb_pattern)
    if idx == -1:
        print(f"  ERROR: heartbeat function not found in {f['file']}: {hb_pattern}")
        return False

    # Find end of heartbeat function (closing brace on its own line or } at end)
    # Search for the closing } of the function
    brace_count = 0
    found_open = False
    end_idx = idx
    for i in range(idx, len(content)):
        if content[i] == '{':
            brace_count += 1
            found_open = True
        elif content[i] == '}':
            brace_count -= 1
            if found_open and brace_count == 0:
                end_idx = i + 1
                break

    # Insert heartbeat_instance right after the heartbeat function
    heartbeat_instance_code = gen_heartbeat_instance(f)
    content = content[:end_idx] + "\n" + heartbeat_instance_code + content[end_idx:]

    # 3. Append instance-level code at end of file
    if f["is_bridge"]:
        instance_code = gen_bridge_instance_code(f)
    else:
        instance_code = gen_nonbridge_instance_code(f)

    # For bridge files, we also need to add health_agent field to the struct
    # But the struct may be in the header or the .c file
    if f["is_bridge"]:
        struct_type = f["struct_type"]
        param_type = f.get("param_type", struct_type)

        # Try to add health_agent to struct in header
        # Check if struct is defined in header
        # For opaque structs defined in .c, modify .c
        # For structs defined in header, modify header
        pass  # We'll handle struct modification separately

    content = content.rstrip() + "\n" + instance_code

    with open(filepath, "w") as fh:
        fh.write(content)

    print(f"  OK {f['file']}")
    return True


def add_health_agent_to_struct(f):
    """Add nimcp_health_agent_t* health_agent to bridge structs."""
    if not f["is_bridge"]:
        return

    struct_type = f["struct_type"]

    # Determine if struct is in header or .c file
    filepath = os.path.join(BASE, f["file"])
    with open(filepath, "r") as fh:
        c_content = fh.read()

    # Check common patterns for struct definition location
    # Inline structs (in .h): struct X { ... };
    # Opaque structs (in .c): struct X { ... };

    # For sleep bridges, the typedef uses handle pattern
    # Check if struct is defined in the .c file
    struct_patterns = [
        f"struct {struct_type.replace('_t', '').replace('struct ', '')}",
    ]

    # Build a list of patterns to search
    if struct_type.startswith("struct "):
        # e.g. "struct introspection_sleep_bridge_struct"
        search_name = struct_type.replace("struct ", "")
        struct_patterns = [f"struct {search_name} {{"]
    else:
        # e.g. "introspection_fep_bridge_t" -> struct name is "introspection_fep_bridge"
        base_name = struct_type.replace("_t", "")
        struct_patterns = [
            f"struct {base_name} {{",
            f"}} {struct_type};",
        ]

    # First check if health_agent already in struct
    if "health_agent" in c_content and "nimcp_health_agent_t* health_agent" in c_content:
        return

    # Try to find struct definition in .c file
    found_in_c = False
    for pat in struct_patterns:
        if pat in c_content:
            found_in_c = True
            # Find closing }; and add health_agent before it
            struct_start = c_content.find(pat)
            # Find the closing };
            brace_count = 0
            in_struct = False
            for i in range(struct_start, len(c_content)):
                if c_content[i] == '{':
                    brace_count += 1
                    in_struct = True
                elif c_content[i] == '}':
                    brace_count -= 1
                    if in_struct and brace_count == 0:
                        # Insert health_agent before closing }
                        insert_point = i
                        indent = "    "
                        field = f"\n{indent}/* Phase 8: Instance-level health agent */\n{indent}nimcp_health_agent_t* health_agent;\n"
                        c_content = c_content[:insert_point] + field + c_content[insert_point:]
                        with open(filepath, "w") as fh:
                            fh.write(c_content)
                        print(f"  STRUCT+ {f['file']} (added health_agent to struct in .c)")
                        return
            break

    if found_in_c:
        return

    # Try header file
    # Build header path from include pattern
    header_map = {
        "introspection_fep_bridge_t": "include/cognitive/introspection/nimcp_introspection_fep_bridge.h",
        "introspection_substrate_bridge_t": "include/cognitive/introspection/nimcp_introspection_substrate_bridge.h",
        "ensemble_pink_bridge_t": "include/cognitive/introspection/nimcp_ensemble_uncertainty_pink_noise_bridge.h",
        "knowledge_fep_bridge_t": "include/cognitive/knowledge/nimcp_knowledge_fep_bridge.h",
    }

    header = header_map.get(struct_type)
    if header:
        hpath = os.path.join(BASE, header)
        with open(hpath, "r") as fh:
            h_content = fh.read()

        if "nimcp_health_agent_t* health_agent" in h_content:
            return

        # Find the struct closing
        for pat in struct_patterns:
            if pat in h_content:
                struct_start = h_content.find(pat)
                brace_count = 0
                in_struct = False
                for i in range(struct_start, len(h_content)):
                    if h_content[i] == '{':
                        brace_count += 1
                        in_struct = True
                    elif h_content[i] == '}':
                        brace_count -= 1
                        if in_struct and brace_count == 0:
                            field = "\n    /* Phase 8: Instance-level health agent */\n    nimcp_health_agent_t* health_agent;\n"
                            h_content = h_content[:i] + field + h_content[i:]
                            with open(hpath, "w") as fh:
                                fh.write(h_content)
                            print(f"  STRUCT+ {header} (added health_agent to struct in header)")
                            return
                break


def main():
    print("Phase 8: Instance-level health agent enhancements")
    print("=" * 60)

    count = 0
    for f in FILES:
        if process_file(f):
            count += 1
            add_health_agent_to_struct(f)

    print(f"\nProcessed {count} / {len(FILES)} files")


if __name__ == "__main__":
    main()
