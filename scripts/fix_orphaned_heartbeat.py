#!/usr/bin/env python3
"""Fix orphaned heartbeat_instance code fragments left by incomplete boilerplate conversion.

Two error patterns:
1. ORPHANED FRAGMENTS: BRIDGE_BOILERPLATE macro followed by stray '}' and
   'if (instance_agent...)' blocks from old heartbeat_instance function body
2. REDEFINITIONS: Manual heartbeat/set_health_agent stubs that duplicate
   what BRIDGE_BOILERPLATE already generates
"""

import re
import sys
import os

ERROR_FILES = [
    "src/cognitive/emotional_tagging/nimcp_emotional_tagging_fep_bridge.c",
    "src/cognitive/emotional_tagging/nimcp_emotional_tagging_substrate_bridge.c",
    "src/cognitive/emotional_tagging/nimcp_emotional_tagging_thalamic_bridge.c",
    "src/cognitive/free_energy/nimcp_fep_immune_bridge.c",
    "src/cognitive/memory/nimcp_working_memory_snn_bridge.c",
    "src/cognitive/meta_learning/nimcp_meta_learning_snn_bridge.c",
    "src/cognitive/meta_learning/nimcp_meta_learning_substrate_bridge.c",
    "src/cognitive/meta_learning/nimcp_meta_learning_thalamic_bridge.c",
    "src/cognitive/neuro_symbolic/bridges/nimcp_hypergraph_kg_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_kg_bridge.c",
    "src/cognitive/omni/bridges/nimcp_omni_wm_logging_bridge.c",
    "src/cognitive/parietal/nimcp_financial_investor_archetype.c",
    "src/cognitive/parietal/nimcp_financial_temporal_credit_bridge.c",
    "src/cognitive/reasoning/nimcp_reasoning_fep_bridge.c",
    "src/cognitive/reasoning/nimcp_reasoning_substrate_bridge.c",
    "src/cognitive/reasoning/nimcp_reasoning_thalamic_bridge.c",
    "src/cognitive/recursive/nimcp_omni_rcog_bridge.c",
    "src/cognitive/salience/nimcp_surprise_amplifier.c",
    "src/cognitive/vae/nimcp_vae_decoder.c",
    "src/cognitive/vae/nimcp_vae_encoder.c",
    "src/cognitive/wellbeing/nimcp_wellbeing_fep_bridge.c",
    "src/cognitive/wellbeing/nimcp_wellbeing_substrate_bridge.c",
    "src/cognitive/wellbeing/nimcp_wellbeing_thalamic_bridge.c",
    "src/cognitive/working_memory/nimcp_working_memory_fep_bridge.c",
    "src/cognitive/working_memory/nimcp_working_memory_sleep_bridge.c",
    "src/cognitive/working_memory/nimcp_working_memory_substrate_bridge.c",
    "src/cognitive/working_memory/nimcp_working_memory_thalamic_bridge.c",
    "src/cognitive/nimcp_meta_learning_fep_bridge.c",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()
    original = content

    # Pattern 1: Remove orphaned heartbeat_instance body fragments
    # These appear as:
    #   BRIDGE_BOILERPLATE(...)\n\n}\n\n\n    if (instance_agent ...) {\n        ...\n    }\n}
    content = re.sub(
        r'(BRIDGE_BOILERPLATE(?:_MESH_ONLY|_MINIMAL)?\([^)]+\))\s*\n'
        r'\s*\}\s*\n'          # stray closing brace
        r'\s*\n*'
        r'\s*if\s*\(instance_agent\s*&&\s*instance_agent\s*!=\s*g_\w+_health_agent\)\s*\{\s*\n'
        r'\s*nimcp_health_agent_heartbeat_ex\(instance_agent,\s*operation,\s*progress\);\s*\n'
        r'\s*\}\s*\n'
        r'\s*\}\s*\n',
        r'\1\n\n',
        content
    )

    # Pattern 1b: Same but with _instance_health_agent instead of _health_agent
    content = re.sub(
        r'(BRIDGE_BOILERPLATE(?:_MESH_ONLY|_MINIMAL)?\([^)]+\))\s*\n'
        r'\s*\}\s*\n'
        r'\s*\n*'
        r'\s*if\s*\(instance_agent\s*&&\s*instance_agent\s*!=\s*g_\w+_instance_health_agent\)\s*\{\s*\n'
        r'\s*nimcp_health_agent_heartbeat_ex\(instance_agent,\s*operation,\s*progress\);\s*\n'
        r'\s*\}\s*\n'
        r'\s*\}\s*\n',
        r'\1\n\n',
        content
    )

    # Pattern 1c: Orphan fragments after alias/stub line
    content = re.sub(
        r'(void\s+\w+_set_health_agent\([^)]*\)\s*\{[^}]*\})\s*\n'
        r'\s*\n*'
        r'\s*\}\s*\n'
        r'\s*\n*'
        r'\s*if\s*\(instance_agent\s*&&\s*instance_agent\s*!=\s*g_\w+_(?:instance_)?health_agent\)\s*\{\s*\n'
        r'\s*nimcp_health_agent_heartbeat_ex\(instance_agent,\s*operation,\s*progress\);\s*\n'
        r'\s*\}\s*\n'
        r'\s*\}\s*\n',
        r'\1\n\n',
        content
    )

    # Pattern 1d: Just orphan }\n with instance_agent block (no preceding BRIDGE_BOILERPLATE on same line)
    content = re.sub(
        r'\n\}\s*\n'
        r'\s*\n*'
        r'\s*if\s*\(instance_agent\s*&&\s*instance_agent\s*!=\s*g_\w+_(?:instance_)?health_agent\)\s*\{\s*\n'
        r'\s*nimcp_health_agent_heartbeat_ex\(instance_agent,\s*operation,\s*progress\);\s*\n'
        r'\s*\}\s*\n'
        r'\s*\}\s*\n',
        r'\n',
        content
    )

    # Pattern 2: Remove manual heartbeat stub that conflicts with BRIDGE_BOILERPLATE
    # Match: static inline void MODULE_heartbeat(const char* op/operation, float progress) { ... }
    # But only if BRIDGE_BOILERPLATE is present (which generates same function)
    if 'BRIDGE_BOILERPLATE' in content:
        # Extract module name from BRIDGE_BOILERPLATE call
        m = re.search(r'BRIDGE_BOILERPLATE(?:_MESH_ONLY|_MINIMAL)?\((\w+)', content)
        if m:
            module = m.group(1)
            # Remove manual heartbeat function (with doc comment if present)
            content = re.sub(
                r'(?:/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n)?'
                r'static\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat\s*\([^)]*\)\s*\{[^}]*\}\s*\n',
                '',
                content,
                flags=re.DOTALL
            )
            # Remove manual heartbeat_instance function
            content = re.sub(
                r'(?:/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n)?'
                r'static\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat_instance\s*\([^)]*\)\s*\{[^}]*\}\s*\n',
                '',
                content,
                flags=re.DOTALL
            )
            # Remove manual set_health_agent_internal stub (different name, OK to keep)
            # But remove set_health_agent that conflicts
            # Only remove if it's a stub (simple body)
            content = re.sub(
                r'/\*\s*Stub[^*]*\*/\s*\n'
                r'static\s+inline\s+void\s+' + re.escape(module) + r'_set_health_agent(?:_internal)?\s*\([^)]*\)\s*\{[^}]*\}\s*\n'
                r'(?:/\*\s*Stub[^*]*\*/\s*\n'
                r'static\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat\s*\([^)]*\)\s*\{[^}]*\}\s*\n)?',
                '',
                content,
                flags=re.DOTALL
            )

    # Clean up excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content == original:
        return False

    with open(filepath, 'w') as f:
        f.write(content)
    return True


def main():
    dry_run = '--dry-run' in sys.argv
    fixed = 0
    for rel_path in ERROR_FILES:
        filepath = os.path.join(ROOT, rel_path)
        if not os.path.exists(filepath):
            print(f"  SKIP (not found): {rel_path}")
            continue
        try:
            if fix_file(filepath):
                fixed += 1
                print(f"  FIXED: {rel_path}")
            else:
                print(f"  NO CHANGE: {rel_path}")
        except Exception as e:
            print(f"  ERROR: {rel_path}: {e}")

    print(f"\nFixed {fixed}/{len(ERROR_FILES)} files")

if __name__ == '__main__':
    main()
