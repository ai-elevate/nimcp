#!/usr/bin/env python3
"""Fix broken files that reference instance_health_agent but lack BRIDGE_BOILERPLATE.

These files had the manual declaration removed but still use NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
instead of BRIDGE_BOILERPLATE. This script upgrades them.

Two categories:
1. BROKEN: Reference g_*_instance_health_agent but don't declare it (compilation error)
2. OLD_PATTERN: Still have complete old boilerplate (mesh register, heartbeat, etc.)
"""

import re
import sys
import os
import subprocess

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

def get_target_files():
    """Find files that use NIMCP_DECLARE_HEALTH_AGENT_ATOMIC + reference instance_health_agent."""
    # Files with old macro AND instance_health_agent references
    result = subprocess.run(
        ['grep', '-rl', 'g_.*_instance_health_agent', SRC_DIR, '--include=*.c'],
        capture_output=True, text=True
    )
    instance_files = set(result.stdout.strip().split('\n'))

    result2 = subprocess.run(
        ['grep', '-rl', 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC', SRC_DIR, '--include=*.c'],
        capture_output=True, text=True
    )
    old_macro_files = set(result2.stdout.strip().split('\n'))

    # Files with old macro that DON'T have BRIDGE_BOILERPLATE
    targets = []
    for f in instance_files & old_macro_files:
        if not f:
            continue
        with open(f) as fh:
            content = fh.read()
        if 'BRIDGE_BOILERPLATE' not in content:
            targets.append(f)
    return sorted(targets)

def convert_file(filepath, dry_run=False):
    with open(filepath) as f:
        content = f.read()

    # Extract module name
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if not m:
        return False, "no module name"
    module_name = m.group(1)

    # Extract mesh category
    m2 = re.search(r'mesh_adapter_get_default_channel\((\w+)\)', content)
    category = m2.group(1) if m2 else 'MESH_ADAPTER_CATEGORY_COGNITIVE'

    # Determine macro variant based on what's in the file
    has_mesh = bool(re.search(r'_mesh_register\s*\(', content))
    has_hb_def = bool(re.search(
        r'(static\s+inline\s+void|void)\s+\w+_heartbeat_instance\s*\(', content
    ))
    # Also check if there are heartbeat_instance CALLS (not just definition)
    has_hb_calls = bool(re.search(r'\w+_heartbeat_instance\s*\(', content))

    if has_mesh or has_hb_def or has_hb_calls:
        macro = f'BRIDGE_BOILERPLATE({module_name}, {category})'
    else:
        macro = f'BRIDGE_BOILERPLATE_MINIMAL({module_name})'

    original = content

    # 1. Replace include
    content = content.replace(
        '#include "utils/fault_tolerance/nimcp_health_agent_macros.h"',
        '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    )

    # 2. Replace NIMCP_DECLARE_HEALTH_AGENT_ATOMIC line with BRIDGE_BOILERPLATE
    # Handle cases with or without trailing comment/newlines
    content = re.sub(
        r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(\w+\)\s*\n',
        macro + '\n',
        content
    )

    # 3. Remove mesh registration block (if present)
    # Try removing the full section with comment headers
    content = re.sub(
        r'//[=\-]+\s*\n'
        r'//\s*Mesh Participant Registration\s*\n'
        r'//[=\-]+\s*\n'
        r'\s*\n'
        r'static\s+mesh_participant_id_t\s+\w+_mesh_id\s*=\s*0;\s*\n'
        r'static\s+mesh_participant_registry_t\*\s+\w+_mesh_registry\s*=\s*NULL;\s*\n'
        r'\s*\n'
        r'(?:static\s+)?nimcp_error_t\s+\w+_mesh_register\s*\([^)]*\)\s*\{[^}]+\}\s*\n'
        r'\s*\n'
        r'(?:static\s+)?void\s+\w+_mesh_unregister\s*\(\s*void\s*\)\s*\{[^}]+\}\s*\n',
        '\n',
        content,
        flags=re.DOTALL
    )

    # Try without comment header
    content = re.sub(
        r'static\s+mesh_participant_id_t\s+\w+_mesh_id\s*=\s*0;\s*\n'
        r'static\s+mesh_participant_registry_t\*\s+\w+_mesh_registry\s*=\s*NULL;\s*\n'
        r'\s*\n'
        r'(?:static\s+)?nimcp_error_t\s+\w+_mesh_register\s*\([^)]*\)\s*\{[^}]+\}\s*\n'
        r'\s*\n'
        r'(?:static\s+)?void\s+\w+_mesh_unregister\s*\(\s*void\s*\)\s*\{[^}]+\}\s*\n',
        '\n',
        content,
        flags=re.DOTALL
    )

    # 4. Remove manual heartbeat_instance function DEFINITION (keep calls)
    content = re.sub(
        r'(?:/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n)?'  # optional /** ... */ doc comment
        r'static\s+inline\s+void\s+\w+_heartbeat_instance\s*\(\s*\n'
        r'\s*nimcp_health_agent_t\*\s+instance_agent,\s*(?:const\s+char\*\s+operation,\s*float\s+progress\s*)?\)\s*\n'
        r'\{[^}]+\}\s*\n',
        '\n',
        content,
        flags=re.DOTALL
    )

    # Also try single-line version
    content = re.sub(
        r'static\s+inline\s+void\s+\w+_heartbeat_instance\s*\([^)]+\)\s*\{[^}]+\}\s*\n',
        '\n',
        content,
        flags=re.DOTALL
    )

    # 5. Remove manual instance_health_agent declaration
    content = re.sub(
        r'static\s+nimcp_health_agent_t\*\s+g_\w+_instance_health_agent\s*=\s*NULL;\s*\n',
        '',
        content
    )

    # 6. Remove empty "Phase 8" comment blocks
    content = re.sub(
        r'/\*\s*=+\s*\n'
        r'\s*\*\s*Phase\s+8[^\n]*Instance-Level Health Agent[^\n]*\n'
        r'\s*\*\s*=+\s*\*/\s*\n\s*\n',
        '\n',
        content
    )

    # 7. Clean up excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content == original:
        return False, "no change"

    if dry_run:
        print(f"  DRY: {filepath} -> {macro}")
        return True, macro

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  OK: {filepath} -> {macro}")
    return True, macro

def main():
    dry_run = '--dry-run' in sys.argv

    files = get_target_files()
    print(f"Found {len(files)} files to fix")

    converted = 0
    for f in files:
        try:
            ok, info = convert_file(f, dry_run)
            if ok:
                converted += 1
        except Exception as e:
            print(f"  ERR: {f}: {e}")

    print(f"\nConverted {converted}/{len(files)} files")

if __name__ == '__main__':
    main()
