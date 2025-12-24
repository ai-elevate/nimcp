#!/usr/bin/env python3
"""
Fix mutex init/destroy pattern for pointer-based mutex in bridge_base_t.

The bridge_base_t has: nimcp_mutex_t* mutex;

So initialization should be:
1. Allocate: bridge->base.mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
2. Init: nimcp_mutex_init(bridge->base.mutex, NULL);

And destruction should be:
1. Destroy: nimcp_mutex_destroy(bridge->base.mutex);
2. Free: nimcp_free(bridge->base.mutex);

This script fixes files that still use the old pattern:
  nimcp_mutex_init(&bridge->base.mutex, NULL)
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

FILES_TO_FIX = [
    'src/plasticity/orchestrator/nimcp_axon_orchestrator_bridge.c',
    'src/plasticity/orchestrator/nimcp_neuron_orchestrator_bridge.c',
    'src/plasticity/orchestrator/nimcp_dendrite_orchestrator_bridge.c',
    'src/perception/cortical/nimcp_audio_cortical_bridge.c',
    'src/perception/cortical/nimcp_speech_cortical_bridge.c',
    'src/perception/cortical/nimcp_visual_cortical_bridge.c',
    'src/cognitive/working_memory/nimcp_working_memory_substrate_bridge.c',
]

def fix_file(filepath):
    """Fix mutex init/destroy patterns in file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Pattern 1: Fix mutex_init with &
    # Change: nimcp_mutex_init(&bridge->base.mutex, NULL)
    # To: Allocate then init
    # This is tricky - we need to insert allocation before init

    # First, fix simple cases where init is on one line
    # Pattern: if (nimcp_mutex_init(&bridge->base.mutex, NULL) == ...
    content = re.sub(
        r'(\s*)/\* Create mutex \*/\s*\n\s*if \(nimcp_mutex_init\(&bridge->base\.mutex, NULL\) == (?:0|NIMCP_SUCCESS)\)',
        r'\1/* Create mutex */\n\1bridge->base.mutex = nimcp_malloc(sizeof(nimcp_mutex_t));\n\1if (bridge->base.mutex && nimcp_mutex_init(bridge->base.mutex, NULL) == 0)',
        content
    )

    # Also handle: if (nimcp_mutex_init(&bridge->base.mutex, NULL) == NIMCP_SUCCESS) {
    content = re.sub(
        r'if \(nimcp_mutex_init\(&bridge->base\.mutex, NULL\) == (?:0|NIMCP_SUCCESS)\)',
        'bridge->base.mutex = nimcp_malloc(sizeof(nimcp_mutex_t));\n    if (bridge->base.mutex && nimcp_mutex_init(bridge->base.mutex, NULL) == 0)',
        content
    )

    # Fix remaining &bridge->base.mutex in init
    content = re.sub(
        r'nimcp_mutex_init\(&bridge->base\.mutex',
        'nimcp_mutex_init(bridge->base.mutex',
        content
    )

    # Pattern 2: Fix mutex_destroy with &
    # Change: nimcp_mutex_destroy(&bridge->base.mutex);
    # To: nimcp_mutex_destroy(bridge->base.mutex); nimcp_free(bridge->base.mutex);
    content = re.sub(
        r'nimcp_mutex_destroy\(&bridge->base\.mutex\);',
        'nimcp_mutex_destroy(bridge->base.mutex);\n        nimcp_free(bridge->base.mutex);\n        bridge->base.mutex = NULL;',
        content
    )

    # Also fix any remaining & patterns
    content = re.sub(
        r'nimcp_mutex_destroy\(&bridge->base\.mutex',
        'nimcp_mutex_destroy(bridge->base.mutex',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("=" * 60)
    print("Fixing mutex init/destroy patterns for pointer-based mutex")
    print("=" * 60)

    fixed_count = 0
    for relpath in FILES_TO_FIX:
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            print(f"Skipping (not found): {relpath}")
            continue
        if fix_file(filepath):
            print(f"Fixed: {relpath}")
            fixed_count += 1
        else:
            print(f"No changes: {relpath}")

    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
