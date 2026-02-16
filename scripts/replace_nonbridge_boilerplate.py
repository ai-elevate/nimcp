#!/usr/bin/env python3
"""
Replace mesh registration boilerplate in non-bridge .c files.

This script finds and replaces the duplicated mesh registration pattern
with the BRIDGE_BOILERPLATE macros from nimcp_bridge_boilerplate.h.
"""

import re
import os
import sys
import subprocess

# Category mapping based on file path
CATEGORY_MAP = [
    ("src/core/brain/subcortical/", "MESH_ADAPTER_CATEGORY_SUBCORTICAL"),
    ("src/cognitive/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/core/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/dragonfly/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/glial/", "MESH_ADAPTER_CATEGORY_GLIAL"),
    ("src/mesh/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/middleware/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/networking/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/plasticity/", "MESH_ADAPTER_CATEGORY_PLASTICITY"),
    ("src/security/", "MESH_ADAPTER_CATEGORY_SECURITY"),
]

def get_category(filepath):
    """Determine mesh adapter category from file path."""
    for prefix, category in CATEGORY_MAP:
        if filepath.startswith(prefix):
            return category
    return "MESH_ADAPTER_CATEGORY_COGNITIVE"


def find_module_name(content):
    """Extract module name from health agent declaration or manual declaration."""
    # Pattern 1: NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(module_name)
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if m:
        return m.group(1), 'atomic'

    # Pattern 2: static nimcp_health_agent_t* g_MODULE_health_agent = NULL;
    m = re.search(r'static\s+nimcp_health_agent_t\s*\*\s*g_(\w+)_health_agent\s*=\s*NULL\s*;', content)
    if m:
        return m.group(1), 'manual'

    return None, None


def has_heartbeat_instance(content, module_name):
    """Check if the file has a heartbeat_instance function for this module."""
    pattern = rf'static\s+inline\s+void\s+{re.escape(module_name)}_heartbeat_instance\s*\('
    return bool(re.search(pattern, content))


def replace_boilerplate(filepath):
    """Replace boilerplate in a single file. Returns (success, message)."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check for static mesh_id - this confirms it's the boilerplate pattern
    if not re.search(r'static\s+mesh_participant_id_t\s+g_\w+_mesh_id\s*=\s*0\s*;', content):
        return False, "No boilerplate mesh_id found"

    module_name, health_type = find_module_name(content)
    if not module_name:
        return False, "Could not determine module name"

    category = get_category(filepath)
    has_heartbeat = has_heartbeat_instance(content, module_name)

    original = content

    # Step 1: Replace health_agent_macros.h include with bridge_boilerplate.h
    # Handle various patterns of this include
    content = content.replace(
        '#include "utils/fault_tolerance/nimcp_health_agent_macros.h"',
        '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    )

    # Step 2: Remove NIMCP_DECLARE_HEALTH_AGENT_ATOMIC line (if present - macro includes it)
    if health_type == 'atomic':
        content = re.sub(
            r'\n?NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(' + re.escape(module_name) + r'\)\s*\n?',
            '\n',
            content
        )

    # Step 2b: Remove manual health agent declaration (if present)
    if health_type == 'manual':
        # Remove "/* Health agent: using pre-existing custom implementation */" comment
        content = re.sub(
            r'/\*\s*Health agent:.*?\*/\s*\n',
            '',
            content
        )
        # Remove "static nimcp_health_agent_t* g_MODULE_health_agent = NULL;"
        content = re.sub(
            r'static\s+nimcp_health_agent_t\s*\*\s*g_' + re.escape(module_name) + r'_health_agent\s*=\s*NULL\s*;\s*\n',
            '',
            content
        )
        # Remove stub heartbeat function if present (the non-instance one)
        content = re.sub(
            r'/\*\s*Stub heartbeat.*?\*/\s*\n'
            r'static\s+inline\s+void\s+' + re.escape(module_name) + r'_heartbeat\s*\([^)]*\)\s*\{[^}]*\}\s*\n?',
            '',
            content
        )

    # Step 3: Remove the mesh registration boilerplate block
    # This includes the comment, static globals, register function, and unregister function

    # Pattern for the full mesh registration block:
    # optional comment header, static mesh_id, static mesh_registry,
    # register function, unregister function
    mesh_pattern = (
        r'(?:'
        r'//=+\n//\s*Mesh Participant Registration\s*\n//=+\s*\n'  # comment header
        r'|'
        r'/\*=+\s*\n\s*\*\s*Mesh.*?=+\s*\*/\s*\n'  # C-style comment header
        r')?'
        r'\s*static\s+mesh_participant_id_t\s+g_' + re.escape(module_name) + r'_mesh_id\s*=\s*0\s*;\s*\n'
        r'\s*static\s+mesh_participant_registry_t\s*\*\s*g_' + re.escape(module_name) + r'_mesh_registry\s*=\s*NULL\s*;\s*\n'
    )
    content = re.sub(mesh_pattern, '', content)

    # Remove the register function
    # Match from function signature to closing brace
    register_pattern = (
        r'\n*nimcp_error_t\s+' + re.escape(module_name) + r'_mesh_register\s*\('
        r'mesh_participant_registry_t\s*\*\s*registry\s*\)\s*\{'
        r'[^}]*\}\s*\n'
    )
    content = re.sub(register_pattern, '\n', content)

    # Remove the unregister function
    unregister_pattern = (
        r'\n*void\s+' + re.escape(module_name) + r'_mesh_unregister\s*\(void\)\s*\{'
        r'[^}]*?'  # non-greedy to match first closing brace
        r'\n\s*\}\s*\n'
    )
    content = re.sub(unregister_pattern, '\n', content)

    # Step 4: Remove heartbeat_instance function if present
    if has_heartbeat:
        # Remove doc comment + function
        heartbeat_pattern = (
            r'(?:/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n)?'  # optional doc comment
            r'(?:\/\*\*\s*@brief.*?\*/\s*\n)?'  # alternative @brief comment
            r'static\s+inline\s+void\s+' + re.escape(module_name) + r'_heartbeat_instance\s*\('
            r'[^)]*\)\s*\{'
            r'[^}]*\}'
            r'\s*\n'
        )
        content = re.sub(heartbeat_pattern, '', content)

    # Step 5: Insert the macro call
    # Find where to insert - after the boilerplate header include and mesh includes
    # The macro should go where the NIMCP_DECLARE_HEALTH_AGENT_ATOMIC was

    # Find the position after #include "mesh/nimcp_mesh_adapter.h"
    insert_after = '#include "utils/bridge/nimcp_bridge_boilerplate.h"'

    if insert_after in content:
        if has_heartbeat:
            macro = f'\nBRIDGE_BOILERPLATE({module_name}, {category})\n'
        else:
            macro = f'\nBRIDGE_BOILERPLATE_MESH_ONLY({module_name}, {category})\n'

        # Insert after the boilerplate include, but after mesh includes if they're right after
        # Find the last mesh include line after the boilerplate include
        bp_idx = content.index(insert_after)
        after_bp = content[bp_idx + len(insert_after):]

        # Skip any mesh includes that follow
        lines_after = after_bp.split('\n')
        skip_count = 0
        for line in lines_after:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('#include "mesh/'):
                skip_count += 1
            else:
                break

        insert_pos = bp_idx + len(insert_after)
        for i in range(skip_count):
            next_newline = content.index('\n', insert_pos) + 1
            insert_pos = next_newline

        content = content[:insert_pos] + macro + content[insert_pos:]
    else:
        # Fallback: insert after #include "mesh/nimcp_mesh_adapter.h"
        mesh_include = '#include "mesh/nimcp_mesh_adapter.h"'
        if mesh_include in content:
            idx = content.index(mesh_include) + len(mesh_include)
            # Skip to end of line
            while idx < len(content) and content[idx] != '\n':
                idx += 1
            if idx < len(content):
                idx += 1  # skip the newline

            if has_heartbeat:
                macro = f'\nBRIDGE_BOILERPLATE({module_name}, {category})\n'
            else:
                macro = f'\nBRIDGE_BOILERPLATE_MESH_ONLY({module_name}, {category})\n'

            content = content[:idx] + macro + content[idx:]
        else:
            return False, "Could not find insertion point"

    # Clean up multiple consecutive blank lines (3+ -> 2)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content == original:
        return False, "No changes made"

    with open(filepath, 'w') as f:
        f.write(content)

    macro_type = "BRIDGE_BOILERPLATE" if has_heartbeat else "BRIDGE_BOILERPLATE_MESH_ONLY"
    return True, f"{macro_type}({module_name}, {category})"


def main():
    os.chdir('/home/bbrelin/nimcp')

    # Get list of files
    result = subprocess.run(
        ['grep', '-rl', 'g_.*_mesh_id = 0', 'src/', '--include=*.c'],
        capture_output=True, text=True
    )
    all_files = sorted(result.stdout.strip().split('\n'))

    # Filter out bridge files
    files = [f for f in all_files if not f.endswith('_bridge.c')]

    print(f"Found {len(files)} non-bridge files to process")

    success_count = 0
    fail_count = 0
    failures = []

    for filepath in files:
        try:
            ok, msg = replace_boilerplate(filepath)
            if ok:
                success_count += 1
                print(f"  OK: {filepath} -> {msg}")
            else:
                fail_count += 1
                failures.append((filepath, msg))
                if '--verbose' in sys.argv:
                    print(f"SKIP: {filepath}: {msg}")
        except Exception as e:
            fail_count += 1
            failures.append((filepath, str(e)))
            print(f"ERROR: {filepath}: {e}")

    print(f"\nResults: {success_count} succeeded, {fail_count} failed/skipped")
    if failures:
        print("\nFailures:")
        for f, msg in failures:
            print(f"  {f}: {msg}")


if __name__ == '__main__':
    main()
