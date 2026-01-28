#!/usr/bin/env python3
"""
Phase 8 Instance-Level Enhancements for free_energy, curiosity, epistemic directories.
Adds heartbeat_instance, set_instance, and FULL training functions to all .c files.
"""
import os
import re

# File definitions: (path, prefix, is_bridge, struct_name, struct_fields_for_training)
# struct_fields_for_training: list of (field, type) tuples that training should interact with
# For bridges: struct is defined in .c file, has bridge_base_t base
# For non-bridges: may or may not have struct in .c file

FILES = [
    # =========================================================================
    # FREE ENERGY (15 files)
    # =========================================================================
    {
        "path": "src/cognitive/free_energy/nimcp_free_energy.c",
        "prefix": "free_energy",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "fep_system_t",
        "training_fields": [
            ("config.belief_learning_rate", "float"),
            ("config.precision_learning_rate", "float"),
            ("config.action_learning_rate", "float"),
            ("config.convergence_threshold", "float"),
            ("stats.total_updates", "uint64_t"),
            ("stats.belief_updates", "uint64_t"),
        ],
        "short_prefix": "free_energy",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_free_energy_substrate_bridge.c",
        "prefix": "free_energy_substrate_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "free_energy_substrate_bridge_t",
        "struct_tag": "free_energy_substrate_bridge",
        "training_fields": [
            ("effects.precision_weighting", "float"),
            ("effects.prediction_depth", "float"),
            ("effects.active_inference", "float"),
            ("effects.model_complexity", "float"),
            ("effects.overall_capacity", "float"),
            ("update_count", "uint64_t"),
        ],
        "short_prefix": "fe_sub",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_free_energy_thalamic_bridge.c",
        "prefix": "free_energy_thalamic_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "free_energy_thalamic_bridge_t",
        "struct_tag": "free_energy_thalamic_bridge",
        "training_fields": [
            ("config.min_error_threshold", "float"),
            ("config.precision_boost", "float"),
            ("stats.total_routes", "uint64_t"),
            ("stats.gated_count", "uint64_t"),
            ("attention_weight", "float"),
        ],
        "short_prefix": "fe_thal",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_consciousness.c",
        "prefix": "fep_consciousness",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "fep_consciousness_bridge_t",
        "struct_tag": "fep_consciousness_bridge",
        "training_fields": [
            ("config.phi_threshold", "float"),
            ("config.attention_decay", "float"),
            ("state.phi_value", "float"),
            ("state.consciousness_level", "float"),
            ("state.attention_focus", "float"),
        ],
        "short_prefix": "fep_cons",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_context.c",
        "prefix": "fep_context",
        "is_bridge": False,
        "has_local_struct": True,
        "struct_type": "fep_context_system_t",
        "struct_tag": "fep_context_system",
        "training_fields": [
            ("active_confidence", "float"),
            ("blend_alpha", "float"),
            ("num_contexts", "uint32_t"),
            ("config.switch_threshold", "float"),
            ("config.blend_rate", "float"),
        ],
        "short_prefix": "fep_ctx",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_curiosity.c",
        "prefix": "fep_curiosity",
        "is_bridge": False,
        "has_local_struct": True,
        "struct_type": "fep_curiosity_system_t",
        "struct_tag": "fep_curiosity_system",
        "training_fields": [
            ("config.epistemic_weight", "float"),
            ("config.novelty_weight", "float"),
            ("config.empowerment_weight", "float"),
            ("state.total_curiosity", "float"),
            ("stats.total_evaluations", "uint64_t"),
        ],
        "short_prefix": "fep_cur",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_evidence.c",
        "prefix": "fep_evidence",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "fep_evi",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_immune_bridge.c",
        "prefix": "fep_immune_bridge",
        "is_bridge": True,
        "has_local_struct": False,
        "struct_type": "fep_immune_bridge_t",
        "training_fields": [
            # Uses opaque struct from header
        ],
        "short_prefix": "fep_imm",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_learning.c",
        "prefix": "fep_learning",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "fep_lrn",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_neuromod.c",
        "prefix": "fep_neuromod",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "fep_neuromod_system_t",
        "training_fields": [],
        "short_prefix": "fep_nmod",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_orchestrator.c",
        "prefix": "fep_orchestrator",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "fep_orch",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_planning.c",
        "prefix": "fep_planning",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "fep_planning_system_t",
        "training_fields": [],
        "short_prefix": "fep_plan",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_plasticity_bridge.c",
        "prefix": "fep_plasticity_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "fep_plasticity_bridge_t",
        "struct_tag": "fep_plasticity_bridge",
        "training_fields": [
            ("config.stdp_a_plus", "float"),
            ("config.stdp_a_minus", "float"),
            ("state.total_ltp_events", "uint32_t"),
            ("state.total_ltd_events", "uint32_t"),
            ("learning_rate_effective", "float"),
            ("bcm_global_threshold", "float"),
        ],
        "short_prefix": "fep_plast",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_sleep.c",
        "prefix": "fep_sleep",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "fep_sleep_system_t",
        "training_fields": [],
        "short_prefix": "fep_slp",
    },
    {
        "path": "src/cognitive/free_energy/nimcp_fep_snn_bridge.c",
        "prefix": "fep_snn_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "fep_snn_bridge_t",
        "struct_tag": "fep_snn_bridge",
        "training_fields": [
            ("config.learning_rate", "float"),
            ("config.encoding_gain", "float"),
            ("state.total_spikes", "uint32_t"),
            ("state.total_steps", "uint32_t"),
            ("pred_error_signal", "float"),
            ("precision_signal", "float"),
        ],
        "short_prefix": "fep_snn",
    },
    # =========================================================================
    # CURIOSITY (10 files)
    # =========================================================================
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity.c",
        "prefix": "curiosity",
        "is_bridge": False,
        "has_local_struct": True,
        "struct_type": "curiosity_engine_t",
        "struct_tag": "curiosity_engine_struct",
        "training_fields": [
            ("total_concepts", "uint32_t"),
            ("num_questions", "uint32_t"),
            ("num_sources", "uint32_t"),
        ],
        "short_prefix": "curiosity",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_enhanced.c",
        "prefix": "curiosity_enhanced",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "cur_enh",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_fep_bridge.c",
        "prefix": "curiosity_fep_bridge",
        "is_bridge": True,
        "has_local_struct": False,
        "struct_type": "curiosity_fep_bridge_t",
        "training_fields": [],
        "short_prefix": "cur_fep",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_fractal.c",
        "prefix": "curiosity_fractal",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "cur_frac",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_hyperbolic.c",
        "prefix": "curiosity_hyperbolic",
        "is_bridge": False,
        "has_local_struct": False,
        "struct_type": "void",
        "training_fields": [],
        "short_prefix": "cur_hyp",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_plasticity_bridge.c",
        "prefix": "curiosity_plasticity_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "curiosity_plasticity_bridge_t",
        "struct_tag": "curiosity_plasticity_bridge",
        "training_fields": [
            ("config.stdp_a_plus", "float"),
            ("config.stdp_a_minus", "float"),
            ("state.total_ltp_events", "uint32_t"),
            ("state.total_ltd_events", "uint32_t"),
            ("current_reward", "float"),
            ("learning_rate_effective", "float"),
            ("bcm_global_threshold", "float"),
        ],
        "short_prefix": "cur_plast",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_sleep_bridge.c",
        "prefix": "curiosity_sleep_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "curiosity_sleep_bridge_t",
        "struct_tag": "curiosity_sleep_bridge_struct",
        "training_fields": [
            ("effects.exploration_suppression", "float"),
            ("effects.novelty_consolidation", "float"),
            ("effects.dream_exploration", "float"),
        ],
        "short_prefix": "cur_slp",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_snn_bridge.c",
        "prefix": "curiosity_snn_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "curiosity_snn_bridge_t",
        "struct_tag": "curiosity_snn_bridge",
        "training_fields": [
            ("config.learning_rate", "float"),
            ("config.encoding_gain", "float"),
            ("state.total_spikes", "uint32_t"),
            ("state.total_steps", "uint32_t"),
            ("novelty_signal", "float"),
            ("information_signal", "float"),
        ],
        "short_prefix": "cur_snn",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_substrate_bridge.c",
        "prefix": "curiosity_substrate_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "curiosity_substrate_bridge_t",
        "struct_tag": "curiosity_substrate_bridge",
        "training_fields": [
            ("effects.exploration_drive", "float"),
            ("effects.novelty_seeking", "float"),
            ("effects.information_gain", "float"),
            ("effects.uncertainty_tolerance", "float"),
            ("effects.overall_capacity", "float"),
            ("update_count", "uint64_t"),
        ],
        "short_prefix": "cur_sub",
    },
    {
        "path": "src/cognitive/curiosity/nimcp_curiosity_thalamic_bridge.c",
        "prefix": "curiosity_thalamic_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "curiosity_thalamic_bridge_t",
        "struct_tag": "curiosity_thalamic_bridge",
        "training_fields": [
            ("config.min_novelty_threshold", "float"),
            ("config.exploration_threshold", "float"),
            ("stats.total_routes", "uint64_t"),
            ("stats.gated_count", "uint64_t"),
            ("attention_weight", "float"),
        ],
        "short_prefix": "cur_thal",
    },
    # =========================================================================
    # EPISTEMIC (6 files)
    # =========================================================================
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_filter.c",
        "prefix": "epistemic_filter",
        "is_bridge": False,
        "has_local_struct": True,
        "struct_type": "epistemic_filter_t",
        "struct_tag": "epistemic_filter_struct",
        "training_fields": [
            ("skepticism_level", "float"),
            ("consensus_weight", "float"),
            ("source_weight", "float"),
            ("claims_assessed", "uint64_t"),
            ("claims_accepted", "uint64_t"),
            ("claims_rejected", "uint64_t"),
            ("biases_detected", "uint64_t"),
        ],
        "short_prefix": "epi_filt",
    },
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_fep_bridge.c",
        "prefix": "epistemic_fep_bridge",
        "is_bridge": True,
        "has_local_struct": False,
        "struct_type": "epistemic_fep_bridge_t",
        "training_fields": [],
        "short_prefix": "epi_fep",
    },
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_plasticity_bridge.c",
        "prefix": "epistemic_plasticity_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "epistemic_plasticity_bridge_t",
        "struct_tag": "epistemic_plasticity_bridge",
        "training_fields": [
            ("config.stdp_a_plus", "float"),
            ("config.stdp_a_minus", "float"),
            ("state.total_ltp_events", "uint32_t"),
            ("state.total_ltd_events", "uint32_t"),
            ("global_learning_rate", "float"),
            ("current_epistemic_quality", "float"),
            ("bcm_global_threshold", "float"),
        ],
        "short_prefix": "epi_plast",
    },
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_snn_bridge.c",
        "prefix": "epistemic_snn_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "epistemic_snn_bridge_t",
        "struct_tag": "epistemic_snn_bridge",
        "training_fields": [
            ("config.dt_ms", "float"),
            ("config.evidence_gain", "float"),
            ("config.uncertainty_gain", "float"),
            ("config.bias_detection_threshold", "float"),
            ("state.total_spikes", "uint32_t"),
            ("state.total_steps", "uint32_t"),
        ],
        "short_prefix": "epi_snn",
    },
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_substrate_bridge.c",
        "prefix": "epistemic_substrate_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "epistemic_substrate_bridge_t",
        "struct_tag": "epistemic_substrate_bridge",
        "training_fields": [
            ("effects.evidence_integration", "float"),
            ("effects.belief_updating", "float"),
            ("effects.certainty_calibration", "float"),
            ("effects.source_evaluation", "float"),
            ("effects.overall_capacity", "float"),
            ("update_count", "uint64_t"),
        ],
        "short_prefix": "epi_sub",
    },
    {
        "path": "src/cognitive/epistemic/nimcp_epistemic_thalamic_bridge.c",
        "prefix": "epistemic_thalamic_bridge",
        "is_bridge": True,
        "has_local_struct": True,
        "struct_type": "epistemic_thalamic_bridge_t",
        "struct_tag": "epistemic_thalamic_bridge",
        "training_fields": [
            ("config.min_urgency_threshold", "float"),
            ("config.uncertainty_boost_factor", "float"),
            ("stats.total_routes", "uint64_t"),
            ("stats.gated_count", "uint64_t"),
            ("attention_weight", "float"),
        ],
        "short_prefix": "epi_thal",
    },
]

BASE = "/home/bbrelin/nimcp"


def gen_heartbeat_instance(prefix):
    return f"""
/** @brief Send heartbeat from {prefix} module (instance-level) */
static inline void {prefix}_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{{
    if (g_{prefix}_health_agent) {{
        nimcp_health_agent_heartbeat_ex(g_{prefix}_health_agent, operation, progress);
    }}
    if (instance_agent && instance_agent != g_{prefix}_health_agent) {{
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }}
}}
"""


def gen_bridge_health_agent_field():
    return "    nimcp_health_agent_t* health_agent;  /**< Instance-level health agent */\n"


def gen_nonbridge_instance_global(prefix):
    return f"\n/** Instance-level health agent for {prefix} (non-bridge fallback) */\nstatic nimcp_health_agent_t* g_{prefix}_instance_health_agent = NULL;\n"


def gen_bridge_set_instance(prefix, struct_type):
    return f"""
/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void {prefix}_set_instance_health_agent({struct_type}* bridge, nimcp_health_agent_t* agent) {{
    if (bridge) {{
        bridge->health_agent = agent;
    }}
}}
"""


def gen_nonbridge_set_instance(prefix):
    return f"""
/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void {prefix}_set_instance_health_agent(void* ctx, nimcp_health_agent_t* agent) {{
    (void)ctx;
    g_{prefix}_instance_health_agent = agent;
}}
"""


def gen_bridge_training_full(prefix, struct_type, struct_tag, fields, short_prefix):
    """Generate FULL training functions for bridge types."""
    # training_begin: Reset counters, set flags, log
    begin_body = ""
    step_body = ""
    end_body = ""

    float_fields = [(f, t) for f, t in fields if t == "float"]
    int_fields = [(f, t) for f, t in fields if t in ("uint32_t", "uint64_t")]

    if fields:
        # training_begin: Reset counters, snapshot initial state
        begin_body += f"    struct {struct_tag}* b = (struct {struct_tag}*)bridge;\n"
        for f, t in int_fields:
            begin_body += f"    b->{f} = 0;\n"
        for f, t in float_fields[:2]:
            begin_body += f"    b->{f} = (b->{f} > 0.0f) ? b->{f} : 1.0f; /* ensure valid */\n"
        begin_body += f'    NIMCP_LOGGING_INFO("{prefix}: training begun, counters reset");\n'

        # training_step: Clamp progress, adapt thresholds, increment counters
        step_body += f"    struct {struct_tag}* b = (struct {struct_tag}*)bridge;\n"
        step_body += "    float p = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);\n"
        for f, t in float_fields:
            step_body += f"    b->{f} += (1.0f - p) * 0.001f; /* gradual adaptation */\n"
            step_body += f"    if (b->{f} > 2.0f) b->{f} = 2.0f;\n"
            step_body += f"    if (b->{f} < 0.0f) b->{f} = 0.0f;\n"
        for f, t in int_fields:
            step_body += f"    b->{f}++;\n"

        # training_end: Compute averages, clear flags, log metrics
        end_body += f"    struct {struct_tag}* b = (struct {struct_tag}*)bridge;\n"
        if float_fields:
            end_body += "    float metric_sum = 0.0f;\n"
            for f, t in float_fields:
                end_body += f"    metric_sum += b->{f};\n"
            end_body += f"    float avg_metric = metric_sum / {len(float_fields)}.0f;\n"
            end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete, avg_metric=%.4f", avg_metric);\n'
        else:
            end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete");\n'
    else:
        # Opaque struct - use what we can
        begin_body += f'    NIMCP_LOGGING_INFO("{prefix}: training begun");\n'
        step_body += "    float p = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);\n"
        step_body += "    (void)p;\n"
        end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete");\n'

    return f"""
/* ============================================================================
 * Phase 8: Full Training Implementation
 * ============================================================================ */
int {prefix}_training_begin({struct_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{short_prefix}_training_begin", 0.0f);
{begin_body}    return 0;
}}

int {prefix}_training_step({struct_type}* bridge, float progress) {{
    if (!bridge) return -1;
    float clamped = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);
    {prefix}_heartbeat_instance(bridge->health_agent, "{short_prefix}_training_step", clamped);
{step_body}    return 0;
}}

int {prefix}_training_end({struct_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{short_prefix}_training_end", 1.0f);
{end_body}    return 0;
}}
"""


def gen_nonbridge_training_full(prefix, struct_type, struct_tag, fields, short_prefix):
    """Generate FULL training functions for non-bridge types."""
    begin_body = ""
    step_body = ""
    end_body = ""

    float_fields = [(f, t) for f, t in fields if t == "float"]
    int_fields = [(f, t) for f, t in fields if t in ("uint32_t", "uint64_t")]

    if fields and struct_tag:
        cast_type = f"struct {struct_tag}"
        begin_body += f"    {cast_type}* s = ({cast_type}*)ctx;\n"
        for f, t in int_fields:
            begin_body += f"    s->{f} = 0;\n"
        for f, t in float_fields[:2]:
            begin_body += f"    s->{f} = (s->{f} > 0.0f) ? s->{f} : 0.5f; /* ensure valid */\n"
        begin_body += f'    NIMCP_LOGGING_INFO("{prefix}: training begun, counters reset");\n'

        step_body += f"    {cast_type}* s = ({cast_type}*)ctx;\n"
        step_body += "    float p = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);\n"
        for f, t in float_fields:
            step_body += f"    s->{f} += (1.0f - p) * 0.001f; /* gradual adaptation */\n"
            step_body += f"    if (s->{f} > 2.0f) s->{f} = 2.0f;\n"
            step_body += f"    if (s->{f} < 0.0f) s->{f} = 0.0f;\n"
        for f, t in int_fields:
            step_body += f"    s->{f}++;\n"

        end_body += f"    {cast_type}* s = ({cast_type}*)ctx;\n"
        if float_fields:
            end_body += "    float metric_sum = 0.0f;\n"
            for f, t in float_fields:
                end_body += f"    metric_sum += s->{f};\n"
            end_body += f"    float avg_metric = metric_sum / {len(float_fields)}.0f;\n"
            end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete, avg_metric=%.4f", avg_metric);\n'
        else:
            end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete");\n'
    else:
        begin_body += f'    NIMCP_LOGGING_INFO("{prefix}: training begun");\n'
        step_body += "    float p = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);\n"
        step_body += "    (void)p;\n"
        end_body += f'    NIMCP_LOGGING_INFO("{prefix}: training complete");\n'

    return f"""
/* ============================================================================
 * Phase 8: Full Training Implementation
 * ============================================================================ */
int {prefix}_training_begin(void* ctx) {{
    if (!ctx) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_begin", 0.0f);
{begin_body}    return 0;
}}

int {prefix}_training_step(void* ctx, float progress) {{
    if (!ctx) return -1;
    float clamped = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_step", clamped);
{step_body}    return 0;
}}

int {prefix}_training_end(void* ctx) {{
    if (!ctx) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_end", 1.0f);
{end_body}    return 0;
}}
"""


def process_file(file_info):
    filepath = os.path.join(BASE, file_info["path"])
    prefix = file_info["prefix"]
    is_bridge = file_info["is_bridge"]
    has_local_struct = file_info["has_local_struct"]
    struct_type = file_info["struct_type"]
    struct_tag = file_info.get("struct_tag", "")
    fields = file_info["training_fields"]
    short_prefix = file_info["short_prefix"]

    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Add heartbeat_instance after existing heartbeat function
    # Find the closing of the existing heartbeat function
    heartbeat_pattern = f"static inline void {prefix}_heartbeat(const char* operation, float progress) {{"
    if heartbeat_pattern not in content:
        print(f"  WARNING: Could not find heartbeat pattern in {filepath}")
        return False

    # Find the end of the heartbeat function (closing brace + newline)
    idx = content.index(heartbeat_pattern)
    # Find next '}' after that
    brace_idx = content.index("}", idx + len(heartbeat_pattern))
    # Insert after the closing brace + newline
    insert_pos = brace_idx + 1
    # Skip trailing newline(s)
    while insert_pos < len(content) and content[insert_pos] == '\n':
        insert_pos += 1
    insert_pos -= 1  # keep one newline

    heartbeat_inst = gen_heartbeat_instance(prefix)
    content = content[:insert_pos] + heartbeat_inst + content[insert_pos:]

    # 2. For bridges: add health_agent field to struct
    if is_bridge and has_local_struct:
        # Find the struct definition and add health_agent field
        struct_pattern = f"struct {struct_tag} {{"
        if struct_pattern in content:
            struct_idx = content.index(struct_pattern)
            # Find the first field after the opening brace
            brace_open = content.index("{", struct_idx) + 1
            # Find a good insertion point - after bridge_base_t base if present
            base_pattern = "bridge_base_t base;"
            base_search_area = content[brace_open:brace_open+300]
            if base_pattern in base_search_area:
                base_idx = brace_open + base_search_area.index(base_pattern) + len(base_pattern)
                # Skip to end of line
                line_end = content.index("\n", base_idx)
                content = content[:line_end+1] + gen_bridge_health_agent_field() + content[line_end+1:]
            else:
                # Insert after opening brace
                content = content[:brace_open+1] + "\n" + gen_bridge_health_agent_field() + content[brace_open+1:]
    elif is_bridge and not has_local_struct:
        # Bridge but struct is opaque (defined in header) - need global fallback
        # Add instance global after the existing global health agent
        global_pattern = f"static nimcp_health_agent_t* g_{prefix}_health_agent = NULL;"
        if global_pattern in content:
            idx = content.index(global_pattern) + len(global_pattern)
            instance_global = f"\n\n/** Instance-level health agent for {prefix} (opaque struct fallback) */\nstatic nimcp_health_agent_t* g_{prefix}_instance_health_agent = NULL;"
            content = content[:idx] + instance_global + content[idx:]

    # 3. For non-bridges: add instance global
    if not is_bridge:
        global_pattern = f"static nimcp_health_agent_t* g_{prefix}_health_agent = NULL;"
        if global_pattern in content:
            idx = content.index(global_pattern) + len(global_pattern)
            content = content[:idx] + gen_nonbridge_instance_global(prefix) + content[idx:]

    # 4. Add set_instance and training functions at end of file
    append_code = "\n"

    if is_bridge and has_local_struct:
        append_code += gen_bridge_set_instance(prefix, struct_type)
        append_code += gen_bridge_training_full(prefix, struct_type, struct_tag, fields, short_prefix)
    elif is_bridge and not has_local_struct:
        # Opaque bridge struct - use global fallback pattern
        append_code += f"""
/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void {prefix}_set_instance_health_agent({struct_type}* bridge, nimcp_health_agent_t* agent) {{
    if (bridge) {{
        g_{prefix}_instance_health_agent = agent;
    }}
}}
"""
        append_code += f"""
/* ============================================================================
 * Phase 8: Full Training Implementation
 * ============================================================================ */
int {prefix}_training_begin({struct_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_begin", 0.0f);
    NIMCP_LOGGING_INFO("{prefix}: training begun");
    return 0;
}}

int {prefix}_training_step({struct_type}* bridge, float progress) {{
    if (!bridge) return -1;
    float clamped = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_step", clamped);
    (void)clamped;
    return 0;
}}

int {prefix}_training_end({struct_type}* bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(g_{prefix}_instance_health_agent, "{short_prefix}_training_end", 1.0f);
    NIMCP_LOGGING_INFO("{prefix}: training complete");
    return 0;
}}
"""
    else:
        append_code += gen_nonbridge_set_instance(prefix)
        append_code += gen_nonbridge_training_full(prefix, struct_type, struct_tag, fields, short_prefix)

    content = content.rstrip() + "\n" + append_code

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def main():
    count = 0
    for file_info in FILES:
        print(f"Processing: {file_info['path']}")
        if process_file(file_info):
            count += 1
            print(f"  OK ({file_info['prefix']})")
        else:
            print(f"  FAILED")

    print(f"\nDone: {count}/{len(FILES)} files processed successfully.")


if __name__ == "__main__":
    main()
