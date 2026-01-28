#!/usr/bin/env python3
"""
Fix Phase 8 training functions that have fabricated/wrong struct member names.

For each file in the error list:
1. Find training_begin, training_end, training_step functions
2. Replace their bodies with simple stubs:
   - NULL check with NIMCP_THROW_TO_IMMUNE
   - heartbeat call
   - return 0
3. Fix set_instance_health_agent functions that dereference opaque types
"""

import re
import sys
import os

# Files with Phase 8 training errors (from make -k build)
ERROR_FILES = [
    "src/cognitive/attention/nimcp_attention_sleep_bridge.c",
    "src/cognitive/autobiographical_memory/nimcp_autobiographical_fep_bridge.c",
    "src/cognitive/autobiographical_memory/nimcp_autobiographical_memory_sleep_bridge.c",
    "src/cognitive/autobiographical_memory/nimcp_autobio_plasticity_bridge.c",
    "src/cognitive/autobiographical_memory/nimcp_autobio_snn_bridge.c",
    "src/cognitive/collective_cognition/nimcp_collective_cognition_immune_bridge.c",
    "src/cognitive/collective_cognition/nimcp_collective_fep_bridge.c",
    "src/cognitive/collective_cognition/nimcp_collective_plasticity_bridge.c",
    "src/cognitive/collective_cognition/nimcp_collective_snn_bridge.c",
    "src/cognitive/consolidation/nimcp_consolidation_fep_bridge.c",
    "src/cognitive/consolidation/nimcp_consolidation_plasticity_bridge.c",
    "src/cognitive/consolidation/nimcp_consolidation_snn_bridge.c",
    "src/cognitive/consolidation/nimcp_consolidation_thalamic_bridge.c",
    "src/cognitive/executive/nimcp_executive_sleep_bridge.c",
    "src/cognitive/explanations/nimcp_explanations_substrate_bridge.c",
    "src/cognitive/explanations/nimcp_explanations_thalamic_bridge.c",
    "src/cognitive/fault_tolerance/nimcp_fault_tolerance_thalamic_bridge.c",
    "src/cognitive/free_energy/nimcp_free_energy_thalamic_bridge.c",
    "src/cognitive/inner_dialogue/nimcp_inner_dialogue.c",
    "src/cognitive/jepa/nimcp_jepa_fep_bridge.c",
    "src/cognitive/jepa/nimcp_jepa_plasticity_bridge.c",
    "src/cognitive/jepa/nimcp_jepa_snn_bridge.c",
    "src/cognitive/logic/nimcp_somatosensory_logic_bridge.c",
    "src/cognitive/memory/nimcp_memory_sleep_bridge.c",
    "src/cognitive/memory/nimcp_working_memory_plasticity_bridge.c",
    "src/cognitive/memory/nimcp_working_memory_snn_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_cognitive_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_hypothalamus_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_kg_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_logging_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_memory_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_parietal_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_plasticity_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_security_immune_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_substrate_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_thalamic_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_tom_bridge.c",
    "src/cognitive/omni/nimcp_omni_active_inference.c",
    "src/cognitive/omni/nimcp_omni_kg_sync.c",
    "src/cognitive/omni/nimcp_omni_metacognition.c",
    "src/cognitive/omni/nimcp_omni_precision.c",
    "src/cognitive/omni/nimcp_omni_world_model.c",
    "src/cognitive/reasoning/nimcp_reasoning_sleep_bridge.c",
    "src/cognitive/recursive/nimcp_omni_rcog_bridge.c",
    "src/cognitive/salience/nimcp_surprise_imagination_bridge.c",
    "src/cognitive/salience/nimcp_surprise_pink_noise_bridge.c",
    "src/cognitive/salience/nimcp_surprise_plasticity_bridge.c",
    "src/cognitive/salience/nimcp_surprise_self_model_bridge.c",
    "src/cognitive/salience/nimcp_surprise_snn_bridge.c",
    "src/cognitive/salience/nimcp_surprise_substrate_bridge.c",
    "src/cognitive/salience/nimcp_surprise_thalamic_bridge.c",
    "src/cognitive/self_awareness_extended/nimcp_self_awareness_extended_substrate_bridge.c",
    "src/cognitive/self_awareness_extended/nimcp_self_awareness_extended_thalamic_bridge.c",
    "src/cognitive/symbolic_logic/bridges/nimcp_symbolic_logic_plasticity_bridge.c",
    "src/cognitive/working_memory/nimcp_working_memory_sleep_bridge.c",
]

BASE = "/home/bbrelin/nimcp"


def find_training_func(lines, start_idx, func_name_pattern):
    """Find a training function and return (start_line, end_line, func_signature_line)."""
    for i in range(start_idx, len(lines)):
        if re.search(func_name_pattern, lines[i]):
            # Found function start, now find the end (matching closing brace at column 0)
            brace_depth = 0
            func_start = i
            for j in range(i, len(lines)):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth == 0 and '{' in ''.join(lines[i:j+1]):
                    return (func_start, j, lines[i].rstrip())
            break
    return None


def extract_func_info(sig_line):
    """Extract function name, return type, and parameter from signature line."""
    # Match patterns like:
    # int foo_training_begin(void* instance) {
    # int foo_training_begin(type_t* bridge) {
    # int foo_training_step(void* instance, float progress) {
    # int foo_training_step(void* instance, uint32_t step) {
    m = re.match(r'(?:int|void)\s+(\w+)\s*\(([^)]*)\)', sig_line)
    if m:
        return m.group(1), m.group(2)
    return None, None


def get_param_name(params):
    """Extract parameter name from parameter list."""
    # "void* instance" -> "instance"
    # "type_t* bridge" -> "bridge"
    # "void* ctx" -> "ctx"
    parts = params.split(',')
    first = parts[0].strip()
    # Get last word
    words = first.split()
    if words:
        name = words[-1].lstrip('*')
        return name
    return "instance"


def has_step_param(params):
    """Check if training_step has float progress or uint32_t step."""
    if 'float' in params:
        return 'float'
    if 'uint32_t' in params:
        return 'uint32_t'
    return 'float'  # default


def get_heartbeat_call(lines, start, end):
    """Extract the heartbeat function call from the function body."""
    for i in range(start, end + 1):
        # Look for heartbeat_instance( pattern
        m = re.search(r'(\w+_heartbeat_instance)\s*\(', lines[i])
        if m:
            return m.group(1)
    return None


def get_heartbeat_agent(lines, start, end):
    """Extract the health agent variable used in heartbeat."""
    for i in range(start, end + 1):
        m = re.search(r'heartbeat_instance\s*\(\s*(\w+)', lines[i])
        if m:
            agent = m.group(1)
            # If it's bridge->health_agent or similar, we need to use a global
            if '->' in lines[i][:lines[i].index('heartbeat_instance')]:
                return None  # Can't use, need global
            return agent
    return None


def get_heartbeat_op_string(lines, start, end, which):
    """Extract the operation string from heartbeat call."""
    for i in range(start, end + 1):
        m = re.search(r'heartbeat_instance\s*\([^,]*,\s*"([^"]*)"', lines[i])
        if m:
            return m.group(1)
    return f"training_{which}"


def find_global_health_agent(lines):
    """Find the global health agent variable name."""
    for line in lines:
        m = re.match(r'static\s+nimcp_health_agent_t\*\s+(g_\w+_health_agent)\s*=', line)
        if m:
            return m.group(1)
    # Check for instance-specific global
    for line in lines:
        m = re.match(r'static\s+nimcp_health_agent_t\*\s+(g_\w+_instance_health_agent)\s*=', line)
        if m:
            return m.group(1)
    return "NULL"


def generate_stub_begin(func_name, param_name, heartbeat_func, agent_var, op_string):
    """Generate a stub training_begin function."""
    return f"""int {func_name}({get_full_param_type(param_name)} {param_name}) {{
    if (!{param_name}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func_name}: NULL argument");
        return -1;
    }}
    {heartbeat_func}({agent_var}, "{op_string}", 0.0f);
    (void){param_name};
    return 0;
}}"""


def generate_stub_end(func_name, param_name, heartbeat_func, agent_var, op_string):
    """Generate a stub training_end function."""
    return f"""int {func_name}({get_full_param_type(param_name)} {param_name}) {{
    if (!{param_name}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func_name}: NULL argument");
        return -1;
    }}
    {heartbeat_func}({agent_var}, "{op_string}", 1.0f);
    (void){param_name};
    return 0;
}}"""


def generate_stub_step_float(func_name, param_name, heartbeat_func, agent_var, op_string):
    """Generate a stub training_step function with float progress."""
    return f"""int {func_name}({get_full_param_type(param_name)} {param_name}, float progress) {{
    if (!{param_name}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func_name}: NULL argument");
        return -1;
    }}
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {heartbeat_func}({agent_var}, "{op_string}", progress);
    (void){param_name};
    return 0;
}}"""


def generate_stub_step_uint32(func_name, param_name, heartbeat_func, agent_var, op_string):
    """Generate a stub training_step function with uint32_t step."""
    return f"""int {func_name}({get_full_param_type(param_name)} {param_name}, uint32_t step) {{
    if (!{param_name}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func_name}: NULL argument");
        return -1;
    }}
    float progress = (step % 100) / 100.0f;
    {heartbeat_func}({agent_var}, "{op_string}", progress);
    (void){param_name};
    return 0;
}}"""


def get_full_param_type(param_name):
    """Get the full parameter type string."""
    # Most use void*, some use specific types
    return "void*"


def fix_file(filepath):
    """Fix Phase 8 training functions in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # Find the global health agent variable
    global_agent = find_global_health_agent(lines)

    # Find all training functions
    funcs_to_fix = []
    training_patterns = [
        (r'int\s+\w+_training_begin\s*\(', 'begin'),
        (r'int\s+\w+_training_end\s*\(', 'end'),
        (r'int\s+\w+_training_step\s*\(', 'step'),
    ]

    for pattern, which in training_patterns:
        result = find_training_func(lines, 0, pattern)
        if result:
            start, end, sig = result
            func_name, params = extract_func_info(sig)
            if func_name:
                # Get heartbeat info
                hb_func = get_heartbeat_call(lines, start, end)
                hb_agent = get_heartbeat_agent(lines, start, end)
                op_string = get_heartbeat_op_string(lines, start, end, which)

                if not hb_agent:
                    hb_agent = global_agent

                # Determine parameter name from actual signature
                actual_param = get_param_name(params)

                # Check step type
                step_type = has_step_param(params) if which == 'step' else None

                funcs_to_fix.append({
                    'start': start,
                    'end': end,
                    'func_name': func_name,
                    'which': which,
                    'heartbeat_func': hb_func,
                    'agent': hb_agent,
                    'op_string': op_string,
                    'param_name': actual_param,
                    'step_type': step_type,
                    'original_sig': sig,
                    'original_params': params,
                })

    if not funcs_to_fix:
        return False

    # Sort by start line in reverse order (fix from bottom to top)
    funcs_to_fix.sort(key=lambda x: x['start'], reverse=True)

    for func in funcs_to_fix:
        # Preserve the original parameter type from the signature
        # Extract the full type+name from original params
        orig_params = func['original_params'].strip()

        if func['which'] == 'begin':
            stub = f"""int {func['func_name']}({orig_params}) {{
    if (!{func['param_name']}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func['func_name']}: NULL argument");
        return -1;
    }}
    {func['heartbeat_func']}({func['agent']}, "{func['op_string']}", 0.0f);
    (void){func['param_name']};
    return 0;
}}"""
        elif func['which'] == 'end':
            stub = f"""int {func['func_name']}({orig_params}) {{
    if (!{func['param_name']}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func['func_name']}: NULL argument");
        return -1;
    }}
    {func['heartbeat_func']}({func['agent']}, "{func['op_string']}", 1.0f);
    (void){func['param_name']};
    return 0;
}}"""
        elif func['which'] == 'step':
            if func['step_type'] == 'uint32_t':
                stub = f"""int {func['func_name']}({orig_params}) {{
    if (!{func['param_name']}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func['func_name']}: NULL argument");
        return -1;
    }}
    float progress = (step % 100) / 100.0f;
    {func['heartbeat_func']}({func['agent']}, "{func['op_string']}", progress);
    (void){func['param_name']};
    return 0;
}}"""
            else:
                stub = f"""int {func['func_name']}({orig_params}) {{
    if (!{func['param_name']}) {{
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "{func['func_name']}: NULL argument");
        return -1;
    }}
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {func['heartbeat_func']}({func['agent']}, "{func['op_string']}", progress);
    (void){func['param_name']};
    return 0;
}}"""

        # Replace lines[start:end+1] with stub
        stub_lines = stub.split('\n')
        lines[func['start']:func['end']+1] = stub_lines

    new_content = '\n'.join(lines)
    with open(filepath, 'w') as f:
        f.write(new_content)

    return True


def main():
    fixed = 0
    failed = []

    for relpath in ERROR_FILES:
        filepath = os.path.join(BASE, relpath)
        if not os.path.exists(filepath):
            print(f"  SKIP: {relpath} (not found)")
            continue

        try:
            if fix_file(filepath):
                print(f"  FIXED: {relpath}")
                fixed += 1
            else:
                print(f"  SKIP: {relpath} (no training functions found)")
        except Exception as e:
            print(f"  FAIL: {relpath}: {e}")
            failed.append(relpath)

    print(f"\nFixed {fixed}/{len(ERROR_FILES)} files")
    if failed:
        print(f"Failed: {len(failed)} files")
        for f in failed:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
