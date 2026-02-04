#!/usr/bin/env python3
"""Follow-up fixes for remaining build errors after fix_build_errors.py.

Issues:
1. Cat1 files lost g_MODULE_health_agent variable when macro was removed - add back
2. Cat2 files: mesh registration code references old name without _instance
3. gpu_init_attempted declared too late in file (after first use)
4. health_metrics_t guard didn't work properly
"""

import re
import os

ROOT = "/home/bbrelin/nimcp"
fixed = 0

def read_file(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

# =============================================================================
# Fix 1: Add back health agent variable declaration to Cat1 files
# These files had NIMCP_DECLARE_HEALTH_AGENT_ATOMIC removed because of
# conflicting heartbeat/set functions, but still need the global variable
# =============================================================================

cat1_files_and_modules = [
    ("src/cognitive/parietal/nimcp_financial_bridge.c", "financial_bridge"),
    ("src/cognitive/parietal/nimcp_financial_autobio_bridge.c", "financial_autobio_bridge"),
    ("src/cognitive/parietal/nimcp_financial_emotion_bridge.c", "financial_emotion_bridge"),
    ("src/cognitive/parietal/nimcp_financial_ethics_bridge.c", "financial_ethics_bridge"),
    ("src/cognitive/parietal/nimcp_financial_explanations_bridge.c", "financial_explanations_bridge"),
    ("src/cognitive/parietal/nimcp_financial_emo_attention_bridge.c", "financial_emo_attention_bridge"),
    ("src/cognitive/parietal/nimcp_financial_mental_health_bridge.c", "financial_mental_health_bridge"),
    ("src/cognitive/parietal/nimcp_financial_tom_bridge.c", "financial_tom_bridge"),
    ("src/cognitive/parietal/nimcp_financial_salience_bridge.c", "financial_salience_bridge"),
    ("src/cognitive/parietal/nimcp_financial_world_model_bridge.c", "financial_world_model_bridge"),
    ("src/cognitive/parietal/nimcp_financial_investment.c", "financial_investment"),
    ("src/cognitive/parietal/nimcp_financial_investor_archetype.c", "fin_arch"),
    ("src/cognitive/parietal/nimcp_financial_curiosity_bridge.c", "fin_curiosity"),
    ("src/cognitive/parietal/nimcp_financial_jepa_bridge.c", "fin_jepa"),
    ("src/cognitive/parietal/nimcp_financial_mammillary_bridge.c", "fin_mammillary"),
    ("src/cognitive/parietal/nimcp_financial_market.c", "fin_mkt"),
    ("src/cognitive/parietal/nimcp_financial_neural_bridge.c", "fin_neural"),
    ("src/cognitive/parietal/nimcp_financial_predictive_bridge.c", "fin_predictive"),
    ("src/cognitive/parietal/nimcp_financial_uncertainty_bridge.c", "fin_uncertainty"),
    ("src/cognitive/parietal/nimcp_financial_working_memory_bridge.c", "fin_wm"),
    ("src/cognitive/salience/nimcp_surprise_amplifier.c", "surprise_amplifier"),
    ("src/cognitive/salience/nimcp_surprise_attention_bridge.c", "surprise_att_bridge"),
    ("src/cognitive/salience/nimcp_surprise_fep_bridge.c", "surprise_fep_bridge"),
    ("src/cognitive/salience/nimcp_surprise_gw_bridge.c", "surprise_gw_bridge"),
    ("src/cognitive/vae/nimcp_vae.c", "vae"),
    ("src/cognitive/vae/nimcp_vae_decoder.c", "vae_decoder"),
    ("src/cognitive/vae/nimcp_vae_encoder.c", "vae_encoder"),
    ("src/core/brain/subcortical/nimcp_basal_ganglia.c", "basal_ganglia"),
]

for rel_path, module in cat1_files_and_modules:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    var_name = f"g_{module}_health_agent"

    # Check if the variable is referenced but not declared as a global
    if var_name in content:
        # Check if it's already declared as a global variable (not inside a function)
        decl_pattern = rf'static\s+(?:_Atomic\()?nimcp_health_agent_t\s*\*\)?\s*{re.escape(var_name)}\s*='
        if not re.search(decl_pattern, content):
            # Need to add the declaration
            var_decl = f"static nimcp_health_agent_t* {var_name} = NULL;\n"

            marker = "/* Health agent: using pre-existing custom implementation */"
            if marker in content:
                content = content.replace(marker, marker + "\n" + var_decl, 1)
            else:
                # Insert after last include
                lines = content.split('\n')
                last_include = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include = i
                lines.insert(last_include + 1, var_decl)
                content = '\n'.join(lines)

            write_file(path, content)
            fixed += 1
            print(f"  Added {var_name} declaration to {rel_path}")

# =============================================================================
# Fix 2: Free energy files - mesh registration uses old names
# The mesh registration code references g_fep_context_health_agent but
# after Cat2 fix, the macro generates g_fep_context_instance_health_agent
# Need to update mesh registration code to not reference the health agent
# at all, OR add an alias
# =============================================================================

# Actually, the mesh registration code doesn't reference the health agent.
# The issue is that the file also has older code outside the mesh registration
# that references g_MODULE_health_agent (without _instance). These were added
# by the old migration script. Let me fix them by replacing the old references.

cat2_fixes = {
    "src/cognitive/free_energy/nimcp_fep_context.c": ("fep_context", "fep_context_instance"),
    "src/cognitive/free_energy/nimcp_fep_curiosity.c": ("fep_curiosity", "fep_curiosity_instance"),
    "src/cognitive/free_energy/nimcp_fep_immune_bridge.c": ("fep_immune_bridge", "fep_immune_bridge_instance"),
    "src/cognitive/free_energy/nimcp_fep_learning.c": ("fep_learning", "fep_learning_instance"),
    "src/cognitive/free_energy/nimcp_free_energy.c": ("free_energy", "free_energy_instance"),
}

for rel_path, (old_prefix, new_prefix) in cat2_fixes.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    old_var = f"g_{old_prefix}_health_agent"
    new_var = f"g_{new_prefix}_health_agent"

    if old_var in content and old_var != new_var:
        content = content.replace(old_var, new_var)

        # Also fix set_health_agent references
        old_set = f"{old_prefix}_set_health_agent"
        new_set = f"{new_prefix}_set_health_agent"
        if old_set in content:
            content = content.replace(old_set, new_set)

        # Fix heartbeat references
        old_hb = f"{old_prefix}_heartbeat"
        new_hb = f"{new_prefix}_heartbeat"
        # Be careful not to replace the _internal variant
        if old_hb in content:
            # Replace only standalone heartbeat, not _internal
            content = re.sub(
                rf'\b{re.escape(old_hb)}\b(?!_internal)',
                new_hb,
                content
            )

        write_file(path, content)
        fixed += 1
        print(f"  Fixed old naming references in {rel_path}")

# =============================================================================
# Fix 3: gpu_init_attempted - move declaration before first use
# =============================================================================

gpu_files = {
    "src/cognitive/curiosity/nimcp_curiosity.c": "g_curiosity_gpu_init_attempted",
    "src/cognitive/jepa/nimcp_jepa_predictor.c": "g_jepa_gpu_init_attempted",
}

for rel_path, var_name in gpu_files.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)
    lines = content.split('\n')

    # Find all declaration lines (static bool var_name = ...)
    decl_indices = []
    first_use = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if f'static bool {var_name}' in stripped or f'static _Bool {var_name}' in stripped:
            decl_indices.append(i)
        elif var_name in stripped and first_use is None:
            first_use = i

    if not decl_indices or first_use is None:
        continue

    # If declaration is AFTER first use, move it
    if decl_indices[0] > first_use:
        # Find a good spot: right after g_*_gpu_init_done or after includes
        done_var = var_name.replace('_attempted', '_done')
        insert_after = None
        for i, line in enumerate(lines):
            if done_var in line and 'static' in line:
                insert_after = i
                break

        if insert_after is None:
            # Find last include
            for i, line in enumerate(lines):
                if line.strip().startswith('#include'):
                    insert_after = i

        if insert_after is not None:
            decl_line = f"static bool {var_name} = false;"
            # Remove ALL existing declarations
            new_lines = []
            for i, line in enumerate(lines):
                if f'static bool {var_name}' not in line.strip():
                    new_lines.append(line)
            # Insert at proper position
            new_lines.insert(insert_after + 1, decl_line)
            content = '\n'.join(new_lines)
            write_file(path, content)
            fixed += 1
            print(f"  Moved {var_name} declaration before first use in {rel_path}")

# =============================================================================
# Fix 4: health_metrics_t - the guard didn't work because the regex didn't match
# Let me look at the actual structure and fix it properly
# =============================================================================

mesh_types_path = os.path.join(ROOT, "include/mesh/nimcp_mesh_types.h")
if os.path.exists(mesh_types_path):
    content = read_file(mesh_types_path)

    # Remove any previous broken guard attempt
    content = content.replace('#ifndef HEALTH_METRICS_DEFINED\n#define HEALTH_METRICS_DEFINED\n', '')
    content = content.replace('\n#endif /* HEALTH_METRICS_DEFINED */', '')

    # Find the health_metrics_t typedef and wrap it properly
    # Pattern: typedef struct health_metrics { ... } health_metrics_t;
    pattern = r'(typedef\s+struct\s+health_metrics\s*\{.*?\}\s*health_metrics_t\s*;)'
    m = re.search(pattern, content, re.DOTALL)
    if m:
        original = m.group(0)
        guarded = f"#ifndef HEALTH_METRICS_DEFINED\n#define HEALTH_METRICS_DEFINED\n{original}\n#endif /* HEALTH_METRICS_DEFINED */"
        content = content.replace(original, guarded)
        write_file(mesh_types_path, content)
        fixed += 1
        print(f"  Re-applied health_metrics_t guard in mesh types header")

# Also check if failure_prediction.c has its own health_metrics_t definition
# that it expects to use
fp_path = os.path.join(ROOT, "src/cognitive/fault_tolerance/nimcp_failure_prediction.c")
if os.path.exists(fp_path):
    content = read_file(fp_path)
    # Check if this file defines its own health_metrics_t
    if 'typedef struct health_metrics' in content:
        # Wrap this one too with the same guard
        pattern = r'(typedef\s+struct\s+health_metrics\s*\{.*?\}\s*health_metrics_t\s*;)'
        m = re.search(pattern, content, re.DOTALL)
        if m:
            original = m.group(0)
            if '#ifndef HEALTH_METRICS_DEFINED' not in content:
                guarded = f"#ifndef HEALTH_METRICS_DEFINED\n#define HEALTH_METRICS_DEFINED\n{original}\n#endif /* HEALTH_METRICS_DEFINED */"
                content = content.replace(original, guarded)
                write_file(fp_path, content)
                fixed += 1
                print(f"  Applied health_metrics_t guard in failure_prediction.c")

print(f"\n=== Total fixes applied: {fixed} ===")
