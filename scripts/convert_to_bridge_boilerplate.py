#!/usr/bin/env python3
"""Convert old boilerplate patterns to BRIDGE_BOILERPLATE macro.

For each file that uses NIMCP_DECLARE_HEALTH_AGENT_ATOMIC but not BRIDGE_BOILERPLATE:
1. Extract module name and mesh category
2. Replace old include with bridge_boilerplate.h
3. Replace old boilerplate block with single BRIDGE_BOILERPLATE() call
4. Remove duplicate instance_health_agent declarations
"""

import re
import sys
import os

def find_files_to_convert(src_dir):
    """Find all .c files that need conversion."""
    import subprocess
    result = subprocess.run(
        ['grep', '-rl', 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC', src_dir, '--include=*.c'],
        capture_output=True, text=True
    )
    candidates = result.stdout.strip().split('\n')

    files = []
    for f in candidates:
        if not f:
            continue
        with open(f) as fh:
            content = fh.read()
        if 'BRIDGE_BOILERPLATE' not in content:
            files.append(f)
    return files

def extract_module_name(content):
    """Extract module name from NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(name)."""
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    return m.group(1) if m else None

def extract_mesh_category(content):
    """Extract mesh category from mesh_adapter_get_default_channel(CATEGORY)."""
    m = re.search(r'mesh_adapter_get_default_channel\((\w+)\)', content)
    return m.group(1) if m else 'MESH_ADAPTER_CATEGORY_COGNITIVE'

def has_heartbeat_instance(content, module_name):
    """Check if file has heartbeat_instance function definition."""
    pattern = rf'{module_name}_heartbeat_instance\s*\('
    # Check for function DEFINITION (not just calls)
    def_pattern = rf'(static\s+inline\s+void|void)\s+{module_name}_heartbeat_instance\s*\('
    return bool(re.search(def_pattern, content))

def has_mesh_registration(content, module_name=None):
    """Check if file has manual mesh registration code."""
    return bool(re.search(r'_mesh_register\s*\(', content))

def convert_file(filepath, dry_run=False):
    """Convert a single file to use BRIDGE_BOILERPLATE."""
    with open(filepath) as f:
        content = f.read()

    module_name = extract_module_name(content)
    if not module_name:
        print(f"  SKIP: No NIMCP_DECLARE_HEALTH_AGENT_ATOMIC found in {filepath}")
        return False

    category = extract_mesh_category(content)
    has_hb = has_heartbeat_instance(content, module_name)
    has_mesh = has_mesh_registration(content, module_name)

    original = content

    # Step 1: Replace the health_agent_macros.h include with bridge_boilerplate.h
    content = content.replace(
        '#include "utils/fault_tolerance/nimcp_health_agent_macros.h"',
        '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    )

    # Step 2: Remove NIMCP_DECLARE_HEALTH_AGENT_ATOMIC line
    content = re.sub(
        r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(\w+\)\s*\n',
        '',
        content
    )

    # Step 3: Remove manual mesh registration block
    # Pattern: static mesh_participant_id_t ... through mesh_unregister function
    mesh_block = re.compile(
        r'//[=\-]+\s*\n'  # separator line
        r'//\s*Mesh Participant Registration\s*\n'  # header
        r'//[=\-]+\s*\n'  # separator line
        r'\s*\n'  # blank line
        r'static\s+mesh_participant_id_t\s+\w+_mesh_id\s*=\s*0;\s*\n'  # id var
        r'static\s+mesh_participant_registry_t\*\s+\w+_mesh_registry\s*=\s*NULL;\s*\n'  # registry var
        r'\s*\n'  # blank line
        r'(?:static\s+)?nimcp_error_t\s+\w+_mesh_register\s*\([^)]*\)\s*\{[^}]+\}\s*\n'  # register func
        r'\s*\n'  # blank line
        r'(?:static\s+)?void\s+\w+_mesh_unregister\s*\(\s*void\s*\)\s*\{[^}]+\}\s*\n',
        re.DOTALL
    )
    content = mesh_block.sub('\n', content)

    # Also try without comment header
    mesh_block2 = re.compile(
        r'static\s+mesh_participant_id_t\s+\w+_mesh_id\s*=\s*0;\s*\n'
        r'static\s+mesh_participant_registry_t\*\s+\w+_mesh_registry\s*=\s*NULL;\s*\n'
        r'\s*\n'
        r'(?:static\s+)?nimcp_error_t\s+\w+_mesh_register\s*\([^)]*\)\s*\{[^}]+\}\s*\n'
        r'\s*\n'
        r'(?:static\s+)?void\s+\w+_mesh_unregister\s*\(\s*void\s*\)\s*\{[^}]+\}\s*\n',
        re.DOTALL
    )
    content = mesh_block2.sub('\n', content)

    # Step 4: Remove manual heartbeat_instance function definition
    hb_pattern = re.compile(
        r'(?:/\*\*.*?@brief.*?heartbeat.*?\*/\s*\n)?'  # optional doc comment
        r'static\s+inline\s+void\s+\w+_heartbeat_instance\s*\([^)]*\)\s*\{[^}]+\}\s*\n',
        re.DOTALL
    )
    content = hb_pattern.sub('\n', content)

    # Step 5: Remove manual instance_health_agent variable declaration
    content = re.sub(
        r'static\s+nimcp_health_agent_t\*\s+g_\w+_instance_health_agent\s*=\s*NULL;\s*\n',
        '',
        content
    )

    # Step 6: Remove "Phase 8 Instance-Level Health Agent Support" comment block if now empty
    content = re.sub(
        r'/\*\s*=+\s*\n'
        r'\s*\*\s*Phase\s+8[^\n]*Instance-Level Health Agent Support[^\n]*\n'
        r'\s*\*\s*=+\s*\*/\s*\n\s*\n',
        '\n',
        content
    )

    # Step 7: Insert BRIDGE_BOILERPLATE after the mesh includes
    # Find the right insertion point - after the mesh includes
    if has_mesh and has_hb:
        macro = f'BRIDGE_BOILERPLATE({module_name}, {category})'
    elif has_mesh:
        macro = f'BRIDGE_BOILERPLATE_MESH_ONLY({module_name}, {category})'
    else:
        macro = f'BRIDGE_BOILERPLATE_MINIMAL({module_name})'

    # Insert after mesh/nimcp_mesh_adapter.h include
    mesh_adapter_include = '#include "mesh/nimcp_mesh_adapter.h"'
    if mesh_adapter_include in content:
        content = content.replace(
            mesh_adapter_include,
            mesh_adapter_include + '\n\n' + macro
        )
    else:
        # Insert after mesh_participant.h
        mesh_participant_include = '#include "mesh/nimcp_mesh_participant.h"'
        if mesh_participant_include in content:
            content = content.replace(
                mesh_participant_include,
                mesh_participant_include + '\n#include "mesh/nimcp_mesh_adapter.h"\n\n' + macro
            )

    # Step 8: Clean up excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content == original:
        print(f"  NO CHANGE: {filepath}")
        return False

    if dry_run:
        print(f"  WOULD CONVERT: {filepath} -> {macro}")
        return True

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  CONVERTED: {filepath} -> {macro}")
    return True

def main():
    dry_run = '--dry-run' in sys.argv
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

    files = find_files_to_convert(src_dir)
    print(f"Found {len(files)} files to convert")

    converted = 0
    failed = 0
    for f in sorted(files):
        try:
            if convert_file(f, dry_run):
                converted += 1
        except Exception as e:
            print(f"  ERROR: {f}: {e}")
            failed += 1

    print(f"\nResults: {converted} converted, {failed} failed, {len(files) - converted - failed} unchanged")

if __name__ == '__main__':
    main()
