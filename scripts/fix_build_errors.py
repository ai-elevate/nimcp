#!/usr/bin/env python3
"""Fix build errors caused by health agent migration and other scripts.

Categories:
1. Heartbeat/set_health_agent conflicts - remove NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
   from files that already have custom implementations
2. _instance_ naming mismatch - fix macro module name to include 'instance'
3. Undeclared globals (g_*_bbb, g_*_immune, g_*_kg_ctx) - add stubs
4. gpu_init_attempted undeclared - add declarations
5. Missing includes (nimcp_mutex_t)
6. health_metrics_t conflict - guard the typedef
7. Other specific fixes (MODULE_NAME, LNN constants, vae args)
"""

import re
import os

ROOT = "/home/bbrelin/nimcp"
stats = {"cat1": 0, "cat2": 0, "cat3": 0, "cat4": 0, "cat5": 0, "cat6": 0, "cat7": 0}

def read_file(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

# =============================================================================
# Category 1: Remove NIMCP_DECLARE_HEALTH_AGENT_ATOMIC from files with
# pre-existing custom heartbeat/set_health_agent functions
# =============================================================================

# Files that have conflicting types for heartbeat or set_health_agent
# These already had custom implementations before the migration
cat1_files = [
    # Financial bridge files with conflicting types
    "src/cognitive/parietal/nimcp_financial_bridge.c",
    "src/cognitive/parietal/nimcp_financial_autobio_bridge.c",
    "src/cognitive/parietal/nimcp_financial_emotion_bridge.c",
    "src/cognitive/parietal/nimcp_financial_ethics_bridge.c",
    "src/cognitive/parietal/nimcp_financial_explanations_bridge.c",
    "src/cognitive/parietal/nimcp_financial_emo_attention_bridge.c",
    "src/cognitive/parietal/nimcp_financial_mental_health_bridge.c",
    "src/cognitive/parietal/nimcp_financial_tom_bridge.c",
    "src/cognitive/parietal/nimcp_financial_salience_bridge.c",
    "src/cognitive/parietal/nimcp_financial_world_model_bridge.c",
    # Files with redefinitions (they have both macro-generated and custom heartbeat)
    "src/cognitive/parietal/nimcp_financial_investment.c",
    "src/cognitive/parietal/nimcp_financial_investor_archetype.c",
    "src/cognitive/parietal/nimcp_financial_curiosity_bridge.c",
    "src/cognitive/parietal/nimcp_financial_jepa_bridge.c",
    "src/cognitive/parietal/nimcp_financial_mammillary_bridge.c",
    "src/cognitive/parietal/nimcp_financial_market.c",
    "src/cognitive/parietal/nimcp_financial_neural_bridge.c",
    "src/cognitive/parietal/nimcp_financial_predictive_bridge.c",
    "src/cognitive/parietal/nimcp_financial_uncertainty_bridge.c",
    "src/cognitive/parietal/nimcp_financial_working_memory_bridge.c",
    # Salience files with conflicting set_health_agent
    "src/cognitive/salience/nimcp_surprise_amplifier.c",
    "src/cognitive/salience/nimcp_surprise_attention_bridge.c",
    "src/cognitive/salience/nimcp_surprise_fep_bridge.c",
    "src/cognitive/salience/nimcp_surprise_gw_bridge.c",
    # VAE files
    "src/cognitive/vae/nimcp_vae.c",
    "src/cognitive/vae/nimcp_vae_decoder.c",
    "src/cognitive/vae/nimcp_vae_encoder.c",
    # Basal ganglia
    "src/core/brain/subcortical/nimcp_basal_ganglia.c",
]

for rel_path in cat1_files:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    # Remove the NIMCP_DECLARE_HEALTH_AGENT_ATOMIC line and its surrounding comments
    # Pattern: optional comment block + the macro line
    new_content = re.sub(
        r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(\w+\)\s*\n',
        '/* Health agent: using pre-existing custom implementation */\n',
        content
    )

    if new_content != content:
        write_file(path, new_content)
        stats["cat1"] += 1
        print(f"  Cat1: Removed macro from {rel_path}")

# =============================================================================
# Category 2: Fix _instance_ naming mismatch in free_energy files
# The migration used NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(fep_context) but code
# references g_fep_context_instance_health_agent. Fix: change macro param.
# =============================================================================

cat2_files = {
    "src/cognitive/free_energy/nimcp_fep_context.c": "fep_context_instance",
    "src/cognitive/free_energy/nimcp_fep_curiosity.c": "fep_curiosity_instance",
    "src/cognitive/free_energy/nimcp_fep_immune_bridge.c": "fep_immune_bridge_instance",
    "src/cognitive/free_energy/nimcp_fep_learning.c": "fep_learning_instance",
    "src/cognitive/free_energy/nimcp_free_energy.c": "free_energy_instance",
}

for rel_path, correct_module in cat2_files.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    # Extract current module name from the macro
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if m:
        old_module = m.group(1)
        if old_module != correct_module:
            new_content = content.replace(
                f'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC({old_module})',
                f'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC({correct_module})'
            )
            # Also fix mesh registration module name references
            new_content = new_content.replace(
                f'g_{old_module}_mesh_id',
                f'g_{correct_module}_mesh_id'
            ).replace(
                f'{old_module}_mesh_register',
                f'{correct_module}_mesh_register'
            ).replace(
                f'{old_module}_mesh_unregister',
                f'{correct_module}_mesh_unregister'
            ).replace(
                f'iface.module_name, "{old_module}"',
                f'iface.module_name, "{correct_module}"'
            ).replace(
                f'config.module_name = "{old_module}"',
                f'config.module_name = "{correct_module}"'
            )
            write_file(path, new_content)
            stats["cat2"] += 1
            print(f"  Cat2: Fixed module name {old_module} -> {correct_module} in {rel_path}")

# =============================================================================
# Category 3: Add stub declarations for undeclared globals (g_*_bbb, g_*_immune, g_*_kg_ctx)
# =============================================================================

cat3_fixes = {
    "src/core/brain/subcortical/nimcp_basal_ganglia.c": [
        ("g_basal_ganglia_bbb", "static void* g_basal_ganglia_bbb = NULL;"),
        ("g_basal_ganglia_kg_ctx", "static void* g_basal_ganglia_kg_ctx = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_investment.c": [
        ("g_fin_investment_immune", "static void* g_fin_investment_immune = NULL;"),
        ("g_fin_investment_bbb", "static void* g_fin_investment_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_investor_archetype.c": [
        ("g_fin_arch_immune", "static void* g_fin_arch_immune = NULL;"),
        ("g_fin_arch_bbb", "static void* g_fin_arch_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_neural_bridge.c": [
        ("g_fin_neural_bridge_immune", "static void* g_fin_neural_bridge_immune = NULL;"),
        ("g_fin_neural_bridge_bbb", "static void* g_fin_neural_bridge_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_mammillary_bridge.c": [
        ("g_fin_mammillary_bridge_immune", "static void* g_fin_mammillary_bridge_immune = NULL;"),
        ("g_fin_mammillary_bridge_bbb", "static void* g_fin_mammillary_bridge_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_predictive_bridge.c": [
        ("g_fin_predictive_bridge_immune", "static void* g_fin_predictive_bridge_immune = NULL;"),
        ("g_fin_predictive_bridge_bbb", "static void* g_fin_predictive_bridge_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_jepa_bridge.c": [
        ("g_fin_jepa_bridge_immune", "static void* g_fin_jepa_bridge_immune = NULL;"),
        ("g_fin_jepa_bridge_bbb", "static void* g_fin_jepa_bridge_bbb = NULL;"),
    ],
    "src/cognitive/parietal/nimcp_financial_working_memory_bridge.c": [
        ("g_fin_wm_bridge_immune", "static void* g_fin_wm_bridge_immune = NULL;"),
        ("g_fin_wm_bridge_bbb", "static void* g_fin_wm_bridge_bbb = NULL;"),
    ],
}

for rel_path, stubs in cat3_fixes.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    additions = []
    for var_name, decl in stubs:
        if var_name in content and decl not in content:
            additions.append(decl)

    if additions:
        # Insert stubs after the HEALTH AGENT INTEGRATION section or after includes
        stub_block = "\n/* Stub declarations for subsystem integration globals */\n"
        stub_block += "\n".join(additions) + "\n"

        # Try to insert after HEALTH AGENT INTEGRATION comment block
        marker = "/* Health agent: using pre-existing custom implementation */"
        if marker in content:
            content = content.replace(marker, marker + "\n" + stub_block, 1)
        else:
            # Try to insert after the mesh registration section
            mesh_marker = "NIMCP_DECLARE_HEALTH_AGENT_ATOMIC"
            if mesh_marker in content:
                # Insert after the macro line
                content = re.sub(
                    r'(NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(\w+\))',
                    r'\1\n' + stub_block,
                    content, count=1
                )
            else:
                # Insert after last #include
                lines = content.split('\n')
                last_include = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include = i
                lines.insert(last_include + 1, stub_block)
                content = '\n'.join(lines)

        write_file(path, content)
        stats["cat3"] += 1
        print(f"  Cat3: Added {len(additions)} stubs to {rel_path}")

# =============================================================================
# Category 4: gpu_init_attempted undeclared
# =============================================================================

cat4_files = {
    "src/cognitive/curiosity/nimcp_curiosity.c": "g_curiosity_gpu_init_attempted",
    "src/cognitive/jepa/nimcp_jepa_predictor.c": "g_jepa_gpu_init_attempted",
}

for rel_path, var_name in cat4_files.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    if var_name in content and f"static bool {var_name}" not in content and f"static _Bool {var_name}" not in content:
        # Find the _done variant to insert near it
        done_variant = var_name.replace("_attempted", "_done")
        decl = f"static bool {var_name} = false;\n"

        if done_variant in content:
            # Insert after the _done declaration
            pattern = rf'(static\s+(?:bool|_Bool|_Atomic\(bool\))\s+{re.escape(done_variant)}\s*=\s*(?:false|0)\s*;)'
            m = re.search(pattern, content)
            if m:
                content = content[:m.end()] + "\n" + decl + content[m.end():]
            else:
                # Fallback: insert after includes
                lines = content.split('\n')
                last_include = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include = i
                lines.insert(last_include + 1, decl)
                content = '\n'.join(lines)
        else:
            # Insert after includes
            lines = content.split('\n')
            last_include = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#include'):
                    last_include = i
            lines.insert(last_include + 1, decl)
            content = '\n'.join(lines)

        write_file(path, content)
        stats["cat4"] += 1
        print(f"  Cat4: Added {var_name} declaration to {rel_path}")

# =============================================================================
# Category 5: Missing includes (nimcp_mutex_t)
# =============================================================================

cat5_files = [
    "src/cognitive/creative/bridges/nimcp_creative_training_bridge.c",
    "src/security/nimcp_encrypted_audit.c",
]

for rel_path in cat5_files:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        continue
    content = read_file(path)

    include_line = '#include "utils/thread/nimcp_thread.h"'
    if 'nimcp_mutex_t' in content and include_line not in content:
        # Insert after last existing include
        lines = content.split('\n')
        last_include = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include = i
        lines.insert(last_include + 1, include_line)
        content = '\n'.join(lines)
        write_file(path, content)
        stats["cat5"] += 1
        print(f"  Cat5: Added thread include to {rel_path}")

# =============================================================================
# Category 6: health_metrics_t conflict in nimcp_mesh_types.h
# The mesh types header redefines health_metrics_t conflicting with existing def
# =============================================================================

mesh_types_path = os.path.join(ROOT, "include/mesh/nimcp_mesh_types.h")
if os.path.exists(mesh_types_path):
    content = read_file(mesh_types_path)
    # Wrap the conflicting typedef with an include guard
    if 'health_metrics_t' in content and '#ifndef HEALTH_METRICS_DEFINED' not in content:
        content = content.replace(
            'typedef struct health_metrics {',
            '#ifndef HEALTH_METRICS_DEFINED\n#define HEALTH_METRICS_DEFINED\ntypedef struct health_metrics {'
        )
        # Find the closing of the typedef
        pattern = r'(typedef struct health_metrics \{[^}]*\}\s*health_metrics_t\s*;)'
        m = re.search(pattern, content, re.DOTALL)
        if m:
            content = content.replace(m.group(0), m.group(0) + '\n#endif /* HEALTH_METRICS_DEFINED */')
            write_file(mesh_types_path, content)
            stats["cat6"] += 1
            print(f"  Cat6: Guarded health_metrics_t in mesh types header")

# =============================================================================
# Category 7a: MODULE_NAME undeclared in nimcp_bio_async_plasticity_bridge.c
# =============================================================================

async_path = os.path.join(ROOT, "src/async/bridges/nimcp_bio_async_plasticity_bridge.c")
if os.path.exists(async_path):
    content = read_file(async_path)
    if 'MODULE_NAME' in content and '#define MODULE_NAME' not in content:
        # Add MODULE_NAME definition after includes
        lines = content.split('\n')
        last_include = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include = i
        lines.insert(last_include + 1, '\n#define MODULE_NAME "bio_async_plasticity_bridge"\n')
        content = '\n'.join(lines)
        write_file(async_path, content)
        stats["cat7"] += 1
        print(f"  Cat7a: Added MODULE_NAME to bio_async_plasticity_bridge")

# =============================================================================
# Category 7b: LNN_TRAINING_DEFAULT_* undeclared
# =============================================================================

lnn_path = os.path.join(ROOT, "src/lnn/nimcp_lnn_training.c")
if os.path.exists(lnn_path):
    content = read_file(lnn_path)
    if 'LNN_TRAINING_DEFAULT_LR' in content and '#define LNN_TRAINING_DEFAULT_LR' not in content:
        defines = """
/* Default training hyperparameters */
#ifndef LNN_TRAINING_DEFAULT_LR
#define LNN_TRAINING_DEFAULT_LR 0.001f
#endif
#ifndef LNN_TRAINING_DEFAULT_WEIGHT_DECAY
#define LNN_TRAINING_DEFAULT_WEIGHT_DECAY 0.0001f
#endif
#ifndef LNN_TRAINING_DEFAULT_BETA1
#define LNN_TRAINING_DEFAULT_BETA1 0.9f
#endif
#ifndef LNN_TRAINING_DEFAULT_BETA2
#define LNN_TRAINING_DEFAULT_BETA2 0.999f
#endif
"""
        lines = content.split('\n')
        last_include = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include = i
        lines.insert(last_include + 1, defines)
        content = '\n'.join(lines)
        write_file(lnn_path, content)
        stats["cat7"] += 1
        print(f"  Cat7b: Added LNN training defaults to lnn_training.c")

# =============================================================================
# Category 7c: VAE heartbeat argument mismatch - vae_heartbeat now takes
# (const char*, float) but calls pass (vae_t*, float) or similar
# The VAE file has pre-existing calls like vae_heartbeat(vae, 0.5f) but the
# macro changed vae_heartbeat to take (const char*, float)
# Since we removed the macro in Cat1, this should be fixed already.
# But if there's still an issue with the calls being incompatible...
# Let's check.
# =============================================================================

# For nimcp_fuzzy_bridge.c and nimcp_information_theory.c, need to check specific errors
# These might be cascading from the mesh_types.h issue

print(f"\n=== Summary ===")
print(f"  Cat1 (heartbeat conflicts): {stats['cat1']} files fixed")
print(f"  Cat2 (instance naming): {stats['cat2']} files fixed")
print(f"  Cat3 (undeclared globals): {stats['cat3']} files fixed")
print(f"  Cat4 (gpu_init_attempted): {stats['cat4']} files fixed")
print(f"  Cat5 (missing includes): {stats['cat5']} files fixed")
print(f"  Cat6 (health_metrics_t): {stats['cat6']} files fixed")
print(f"  Cat7 (other): {stats['cat7']} files fixed")
print(f"  Total: {sum(stats.values())} files fixed")
