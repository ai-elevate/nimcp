#!/usr/bin/env python3
"""Fix skipped modules by adding inbox processing with correct variable names."""

import re
import os

# Module configurations: (file, bio_ctx_var, bio_enabled_var, function_name, function_sig_pattern)
MODULES = [
    # No bio_ctx found - use correct variable names
    ("src/cognitive/personality/nimcp_personality.c", "personality_bio_ctx", None, "personality_compute_modifiers", r"void personality_compute_modifiers\(personality_profile_t\* profile\)"),
    ("src/cognitive/wellbeing/nimcp_wellbeing.c", "g_wellbeing_bio_ctx", "g_wellbeing_bio_initialized", "wellbeing_evaluate", r"float wellbeing_evaluate\("),
    ("src/core/axon/nimcp_axon.c", "bio_ctx", None, "axon_step", r"void axon_step\(axon_t\* axon"),
    ("src/core/brain/nimcp_brain_resize.c", "brain->bio_ctx", "brain->bio_async_enabled", "brain_resize", r"nimcp_error_t brain_resize\("),
    ("src/core/brain/nimcp_distributed_cow.c", "brain->bio_ctx", "brain->bio_async_enabled", "distributed_cow_sync", r"nimcp_error_t distributed_cow_sync\("),
    ("src/core/brain_oscillations/nimcp_brain_oscillations.c", "bio_ctx", None, "brain_oscillations_update", r"void brain_oscillations_update\("),
    ("src/core/brain_regions/nimcp_brain_regions.c", "g_brain_regions_bio_ctx", "g_brain_regions_bio_initialized", "brain_regions_update", r"void brain_regions_update\("),
    ("src/core/cortical_columns/nimcp_columnar_connectivity.c", "g_connectivity_bio_ctx", None, "columnar_connectivity_update", r"void columnar_connectivity_update\("),
    ("src/core/cortical_columns/nimcp_cortical_column.c", "g_column_bio_ctx", None, "cortical_column_step", r"void cortical_column_step\("),
    ("src/core/cortical_columns/nimcp_cortical_layers.c", "g_layers_bio_ctx", None, "cortical_layers_update", r"void cortical_layers_update\("),
    ("src/core/cortical_columns/nimcp_feature_hypercolumns.c", "g_hypercolumn_bio_ctx", None, "feature_hypercolumn_process", r"void feature_hypercolumn_process\("),
    ("src/core/cortical_columns/nimcp_orientation_columns.c", "g_orientation_bio_ctx", None, "orientation_column_process", r"void orientation_column_process\("),
    ("src/core/cortical_columns/nimcp_topographic_maps.c", "g_topographic_bio_ctx", None, "topographic_map_update", r"void topographic_map_update\("),
    ("src/core/neuron_models/nimcp_izhikevich.c", "g_izhikevich_bio_ctx", None, "izhikevich_step", r"void izhikevich_step\("),
    ("src/core/neuron_models/nimcp_neuron_model.c", "g_neuron_model_bio_ctx", None, "neuron_model_step", r"void neuron_model_step\("),
    ("src/core/neuron_models/nimcp_two_compartment.c", "g_two_comp_bio_ctx", None, "two_compartment_step", r"void two_compartment_step\("),
    ("src/core/topology/nimcp_community_detection.c", "g_community_bio_ctx", None, "community_detection_run", r"void community_detection_run\("),
    ("src/core/topology/nimcp_fractal_topology.c", "g_fractal_bio_ctx", None, "fractal_topology_update", r"void fractal_topology_update\("),
    ("src/core/topology/nimcp_network_builder.c", "g_builder_bio_ctx", None, "network_builder_step", r"void network_builder_step\("),
    ("src/information/nimcp_shannon.c", "g_bio_ctx", None, "shannon_compute_entropy", r"float shannon_compute_entropy\("),

    # No suitable function - find alternative functions
    ("src/cognitive/analysis/nimcp_network_analysis.c", "g_analysis_bio_ctx", None, "network_analysis_run", r"void network_analysis_run\("),
    ("src/cognitive/autobiographical_memory/nimcp_autobiographical_memory.c", "g_autobio_bio_ctx", None, "autobiographical_memory_recall", r".*autobiographical_memory_recall\("),
    ("src/cognitive/empathetic_response/nimcp_empathetic_response.c", "g_empathy_bio_ctx", None, "empathetic_response_generate", r".*empathetic_response_generate\("),
    ("src/cognitive/epistemic/nimcp_epistemic_filter.c", "g_epistemic_bio_ctx", None, "epistemic_filter_evaluate", r".*epistemic_filter_evaluate\("),
    ("src/cognitive/explanations/nimcp_explanations.c", "g_explanations_bio_ctx", None, "explanation_generate", r".*explanation_generate\("),
    ("src/cognitive/fault_tolerance/nimcp_failure_prediction.c", "g_failure_bio_ctx", None, "failure_prediction_evaluate", r".*failure_prediction_evaluate\("),
    ("src/cognitive/fault_tolerance/nimcp_fault_attention.c", "g_fault_attention_bio_ctx", None, "fault_attention_update", r".*fault_attention_update\("),
    ("src/cognitive/fault_tolerance/nimcp_recovery_episodic_memory.c", "g_episodic_bio_ctx", None, "recovery_episodic_update", r".*recovery_episodic.*update\("),
    ("src/cognitive/fault_tolerance/nimcp_recovery_executive.c", "g_exec_bio_ctx", None, "recovery_executive_update", r".*recovery_executive.*update\("),
    ("src/cognitive/logic/nimcp_symbolic_logic.c", "g_logic_bio_ctx", None, "symbolic_logic_evaluate", r".*symbolic_logic_evaluate\("),
    ("src/cognitive/memory/nimcp_semantic_memory.c", "g_semantic_bio_ctx", None, "semantic_memory_query", r".*semantic_memory_query\("),
    ("src/cognitive/predictive/nimcp_predictive.c", "g_predictive_bio_ctx", None, "predictive_update", r".*predictive.*update\("),
    ("src/cognitive/reasoning/integration/nimcp_reasoning_attention.c", "g_reasoning_attn_bio_ctx", None, "reasoning_attention_focus", r".*reasoning_attention.*\("),
    ("src/cognitive/reasoning/integration/nimcp_reasoning_curiosity.c", "g_reasoning_curiosity_bio_ctx", None, "reasoning_curiosity_evaluate", r".*reasoning_curiosity.*\("),
    ("src/cognitive/reasoning/nimcp_knowledge_base_interface.c", "g_kb_bio_ctx", None, "knowledge_base_query", r".*knowledge_base_query\("),
    ("src/cognitive/reasoning/nimcp_reasoning_factory.c", "g_reasoning_factory_bio_ctx", None, "reasoning_factory_create", r".*reasoning_factory_create\("),
    ("src/cognitive/reasoning/nimcp_reasoning_integration.c", "g_reasoning_int_bio_ctx", None, "reasoning_integration_run", r".*reasoning_integration.*\("),
    ("src/cognitive/self_awareness/nimcp_self_awareness_extended.c", "g_self_awareness_bio_ctx", None, "self_awareness_update", r".*self_awareness.*update\("),
    ("src/cognitive/self_model/nimcp_self_model.c", "g_self_model_bio_ctx", None, "self_model_update", r".*self_model.*update\("),
    ("src/cognitive/sleep_wake/nimcp_sleep_wake.c", "g_sleep_bio_ctx", None, "sleep_wake_transition", r".*sleep_wake.*\("),
    ("src/cognitive/theory_of_mind/nimcp_theory_of_mind.c", "g_tom_bio_ctx", None, "theory_of_mind_infer", r".*theory_of_mind.*\("),
    ("src/gpu/neuron/nimcp_gpu_neuron.c", "g_gpu_neuron_bio_ctx", None, "gpu_neuron_step", r".*gpu_neuron.*step\("),
    ("src/gpu/spike_event/nimcp_spike_event.c", "g_spike_event_bio_ctx", None, "spike_event_process", r".*spike_event.*process\("),
    ("src/middleware/events/nimcp_event_bus.c", "g_event_bus_bio_ctx", None, "event_bus_dispatch", r".*event_bus_dispatch\("),
    ("src/middleware/integration/nimcp_executive_middleware_adapter.c", "g_exec_adapter_bio_ctx", None, "executive_middleware_process", r".*executive_middleware.*\("),
    ("src/middleware/integration/nimcp_middleware_controller.c", "g_controller_bio_ctx", None, "middleware_controller_step", r".*middleware_controller.*\("),
    ("src/middleware/integration/nimcp_quantum_command_propagator.c", "g_quantum_bio_ctx", None, "quantum_command_propagate", r".*quantum.*propagate\("),
    ("src/middleware/pipeline/nimcp_middleware_pipeline.c", "g_pipeline_bio_ctx", None, "middleware_pipeline_run", r".*middleware_pipeline.*run\("),
    ("src/middleware/training/nimcp_regularization.c", "g_regularization_bio_ctx", None, "regularization_apply", r".*regularization_apply\("),
    ("src/networking/distributed/nimcp_distributed_cognition.c", "g_distributed_bio_ctx", None, "distributed_cognition_sync", r".*distributed_cognition.*\("),
    ("src/networking/replication/nimcp_replication.c", "g_replication_bio_ctx", None, "replication_sync", r".*replication.*sync\("),
    ("src/nlp/nimcp_nlp.c", "g_nlp_bio_ctx", None, "nlp_process", r".*nlp_process\("),
]

def find_bio_ctx_var(content):
    """Find the actual bio_ctx variable name in the file."""
    patterns = [
        r'static\s+bio_module_context_t\s+(\w+)\s*=',
        r'(g_\w*bio\w*ctx)\s*=',
        r'(\w+)->bio_ctx\s*=\s*bio_router_register',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None

def find_bio_enabled_var(content, bio_ctx_var):
    """Find the bio_async_enabled variable."""
    if '->' in bio_ctx_var:
        # Struct access pattern
        base = bio_ctx_var.split('->')[0]
        return f"{base}->bio_async_enabled"
    # Look for corresponding initialized flag
    patterns = [
        r'static\s+bool\s+(g_\w*bio\w*initialized)\s*=',
        r'static\s+bool\s+(\w+_bio_initialized)\s*=',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None

def find_suitable_function(content, file_path):
    """Find a suitable function to add inbox processing."""
    basename = os.path.basename(file_path).replace('nimcp_', '').replace('.c', '')

    # Look for common function patterns
    patterns = [
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_update)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_step)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_process)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_run)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_compute)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_evaluate)\s*\([^)]*\)\s*\{{',
        rf'(void|int|bool|nimcp_error_t)\s+({basename}_tick)\s*\([^)]*\)\s*\{{',
        r'(void|int|bool|nimcp_error_t)\s+(\w+_update)\s*\([^)]*\)\s*\{',
        r'(void|int|bool|nimcp_error_t)\s+(\w+_step)\s*\([^)]*\)\s*\{',
        r'(void|int|bool|nimcp_error_t)\s+(\w+_process)\s*\([^)]*\)\s*\{',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(2)
    return None

def add_inbox_processing(file_path):
    """Add inbox processing to a module."""
    full_path = f"/home/bbrelin/nimcp/{file_path}"

    if not os.path.exists(full_path):
        return f"SKIP: {file_path} - File not found"

    with open(full_path, 'r') as f:
        content = f.read()

    # Check if already has inbox processing
    if 'bio_router_process_inbox' in content:
        return f"SKIP: {file_path} - Already has inbox processing"

    # Check if registers with bio-router
    if 'bio_router_register_module' not in content:
        return f"SKIP: {file_path} - No bio-router registration"

    # Find bio_ctx variable
    bio_ctx_var = find_bio_ctx_var(content)
    if not bio_ctx_var:
        return f"SKIP: {file_path} - Could not find bio_ctx variable"

    # Find bio_enabled variable
    bio_enabled_var = find_bio_enabled_var(content, bio_ctx_var)

    # Find suitable function
    func_name = find_suitable_function(content, file_path)
    if not func_name:
        return f"SKIP: {file_path} - No suitable function found"

    # Find the function and add inbox processing after the opening brace
    # Match the function signature and opening brace
    func_pattern = rf'({func_name}\s*\([^)]*\)\s*\{{)'
    match = re.search(func_pattern, content)
    if not match:
        return f"SKIP: {file_path} - Could not locate function {func_name}"

    # Build the inbox processing code
    if bio_enabled_var:
        inbox_code = f"""
    // Process pending bio-async messages
    if ({bio_enabled_var} && {bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
"""
    else:
        inbox_code = f"""
    // Process pending bio-async messages
    if ({bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
"""

    # Insert after the opening brace
    new_content = content[:match.end()] + inbox_code + content[match.end():]

    with open(full_path, 'w') as f:
        f.write(new_content)

    return f"OK:   {file_path} - Added to {func_name}"

def main():
    results = {'ok': 0, 'skip': 0}
    skipped = []

    # Get list of files that register but don't process inbox
    import subprocess

    # Find files with registration
    reg_result = subprocess.run(
        ['grep', '-rl', 'bio_router_register_module', '/home/bbrelin/nimcp/src'],
        capture_output=True, text=True
    )
    registered_files = set(reg_result.stdout.strip().split('\n'))

    # Find files already processing inbox
    inbox_result = subprocess.run(
        ['grep', '-rl', 'bio_router_process_inbox', '/home/bbrelin/nimcp/src'],
        capture_output=True, text=True
    )
    inbox_files = set(inbox_result.stdout.strip().split('\n')) if inbox_result.stdout.strip() else set()

    # Files needing inbox processing
    need_inbox = registered_files - inbox_files

    print(f"Found {len(need_inbox)} files needing inbox processing")
    print("=" * 60)

    for full_path in sorted(need_inbox):
        if not full_path or not full_path.endswith('.c'):
            continue
        file_path = full_path.replace('/home/bbrelin/nimcp/', '')
        result = add_inbox_processing(file_path)
        print(result)
        if result.startswith('OK'):
            results['ok'] += 1
        else:
            results['skip'] += 1
            skipped.append(result)

    print("=" * 60)
    print(f"Modified: {results['ok']}, Skipped: {results['skip']}")

if __name__ == '__main__':
    main()
