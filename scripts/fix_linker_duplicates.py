#!/usr/bin/env python3
"""Fix multiple definition linker errors from duplicate modules.

Multiple source files define the same non-static set_health_agent and
mesh registration functions. Fix by making them static in the secondary files.
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

# Secondary files (the ones with "multiple definition of" error)
# These need their non-static functions made static
secondary_files = [
    "src/cognitive/memory/nimcp_systems_consolidation.c",      # consolidation_*
    "src/cognitive/nimcp_emotional_tagging.c",                  # emotional_tagging_*
    "src/cognitive/parietal/nimcp_counterfactual.c",            # counterfactual_*
    "src/cognitive/parietal/nimcp_parietal_quantum_bridge.c",   # parietal_quantum_bridge_*
    "src/core/brain/regions/broca/nimcp_language_production_bridge.c",  # language_production_bridge_*
    "src/core/directives/nimcp_combinatorial_harm.c",          # combinatorial_harm_*
    "src/dragonfly/nimcp_dragonfly.c",                         # dragonfly_*
    "src/glial/immune/nimcp_astrocyte_immune_bridge.c",        # astrocyte_immune_bridge_*
    "src/integration/adapters/memory/nimcp_hippocampus_adapter.c",  # hippocampus_adapter_*
    "src/lnn/nimcp_lnn_training.c",                            # lnn_*
    "src/middleware/features/nimcp_feature_extractor.c",        # feature_extractor_*
    "src/networking/nlp/nimcp_nlp.c",                          # nlp_*
    "src/security/nimcp_capability_control.c",                 # capability_*
]

for rel_path in secondary_files:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        print(f"  SKIP: {rel_path} not found")
        continue

    content = read_file(path)
    changed = False

    # 1. Replace NIMCP_DECLARE_HEALTH_AGENT_ATOMIC with NIMCP_DECLARE_HEALTH_AGENT_STATIC
    #    (static version makes set_health_agent static)
    if 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(' in content:
        content = content.replace(
            'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(',
            'NIMCP_DECLARE_HEALTH_AGENT_STATIC('
        )
        changed = True
    elif 'NIMCP_DECLARE_HEALTH_AGENT(' in content:
        content = content.replace(
            'NIMCP_DECLARE_HEALTH_AGENT(',
            'NIMCP_DECLARE_HEALTH_AGENT_STATIC('
        )
        changed = True

    # 2. Make mesh_register and mesh_unregister functions static
    # Pattern: nimcp_error_t MODULE_mesh_register(...)
    content = re.sub(
        r'^(nimcp_error_t\s+\w+_mesh_register\s*\()',
        r'static \1',
        content,
        flags=re.MULTILINE
    )

    # Pattern: void MODULE_mesh_unregister(...)
    content = re.sub(
        r'^(void\s+\w+_mesh_unregister\s*\()',
        r'static \1',
        content,
        flags=re.MULTILINE
    )

    # Avoid double-static (in case it was already static)
    content = content.replace('static static ', 'static ')

    if changed or 'mesh_register' in content:
        write_file(path, content)
        fixed += 1
        print(f"  Fixed: {rel_path}")

print(f"\n=== {fixed} files fixed ===")
