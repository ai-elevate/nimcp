#!/usr/bin/env python3
"""
Replace bridge boilerplate in src/core/ files with BRIDGE_BOILERPLATE_MESH_ONLY macro.

This script finds the mesh registration boilerplate pattern in each file and replaces it
with a single macro call from nimcp_bridge_boilerplate.h.

Pattern being replaced:
  - NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(module) declaration
  - static mesh_participant_id_t g_xxx_mesh_id = 0;
  - static mesh_participant_registry_t* g_xxx_mesh_registry = NULL;
  - nimcp_error_t xxx_mesh_register(mesh_participant_registry_t* registry) { ... }
  - void xxx_mesh_unregister(void) { ... }

Replaced with:
  BRIDGE_BOILERPLATE_MESH_ONLY(module, CATEGORY)
"""

import re
import os
import sys


def find_files():
    """Find all files in src/core/ that have the boilerplate pattern."""
    import subprocess
    # Find files that have NIMCP_DECLARE_HEALTH_AGENT_ATOMIC (the boilerplate marker)
    result = subprocess.run(
        ['grep', '-rl', 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC', '/home/bbrelin/nimcp/src/core'],
        capture_output=True, text=True
    )
    candidates = sorted(result.stdout.strip().split('\n'))

    # Filter to only those that also have mesh_participant_register (function definition, not just call)
    files = []
    for f in candidates:
        if not f:
            continue
        with open(f, 'r') as fh:
            content = fh.read()
        # Must have the register function DEFINITION (not just a call)
        if '_mesh_register(mesh_participant_registry_t' in content:
            # Skip if already processed
            if 'BRIDGE_BOILERPLATE_MESH_ONLY' in content or 'BRIDGE_BOILERPLATE(' in content:
                continue
            # Skip files with static mesh functions (non-standard pattern)
            if 'static nimcp_error_t' in content and '_mesh_register' in content:
                # Check if the register function itself is static
                if re.search(r'static\s+nimcp_error_t\s+\w+_mesh_register', content):
                    print(f"  SKIP (static linkage): {f}")
                    continue
            files.append(f)
    return files


def get_category(filepath):
    """Determine the mesh adapter category from the file path."""
    if '/subcortical/' in filepath:
        return 'MESH_ADAPTER_CATEGORY_SUBCORTICAL'
    else:
        return 'MESH_ADAPTER_CATEGORY_COGNITIVE'


def process_file(filepath):
    """Process a single file, replacing boilerplate with macro."""
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find the module name from NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(xxx)
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if not m:
        print(f"  SKIP: No NIMCP_DECLARE_HEALTH_AGENT_ATOMIC found in {filepath}")
        return False
    module_name = m.group(1)

    category = get_category(filepath)

    # Find the line with NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
    health_agent_line = None
    for i, line in enumerate(lines):
        if 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(' in line:
            health_agent_line = i
            break

    if health_agent_line is None:
        print(f"  SKIP: Could not find HEALTH_AGENT line in {filepath}")
        return False

    # Scan backwards from health_agent_line to find the start of the boilerplate includes
    bp_start = health_agent_line

    for i in range(health_agent_line - 1, -1, -1):
        line = lines[i].strip()
        if line == '' or line.startswith('//='):
            bp_start = i
            continue
        if '#include' in line:
            # Check if this is a boilerplate include
            if any(h in line for h in [
                'nimcp_health_agent_macros.h',
                'nimcp_mesh_participant.h',
                'nimcp_mesh_adapter.h',
                'nimcp_logging.h',
                '<stddef.h>',
                'nimcp_bridge_boilerplate.h',
            ]):
                bp_start = i
                continue
            else:
                break
        else:
            break

    # Now find the end of the unregister function
    unregister_pattern = f'{module_name}_mesh_unregister'
    unregister_start = None
    for i in range(health_agent_line, len(lines)):
        if unregister_pattern in lines[i] and ('void' in lines[i] or unregister_pattern + '(' in lines[i]):
            unregister_start = i
            break

    if unregister_start is None:
        # Try a more general pattern - sometimes the function name doesn't match the module name
        for i in range(health_agent_line, len(lines)):
            if '_mesh_unregister' in lines[i] and 'void' in lines[i]:
                unregister_start = i
                break

    if unregister_start is None:
        print(f"  SKIP: Could not find unregister function in {filepath}")
        return False

    # Find the closing brace of the unregister function
    brace_depth = 0
    unregister_end = None
    in_func = False
    for i in range(unregister_start, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                brace_depth += 1
                in_func = True
            elif ch == '}':
                brace_depth -= 1
                if in_func and brace_depth == 0:
                    unregister_end = i
                    break
        if unregister_end is not None:
            break

    if unregister_end is None:
        print(f"  SKIP: Could not find end of unregister function in {filepath}")
        return False

    # Skip any trailing blank lines after the unregister end
    bp_end = unregister_end
    while bp_end + 1 < len(lines) and lines[bp_end + 1].strip() == '':
        bp_end += 1

    # Check which includes we need to preserve/add
    initial_includes = '\n'.join(lines[:bp_start])
    has_stddef = '<stddef.h>' in initial_includes
    has_logging = 'nimcp_logging.h' in initial_includes
    has_health_agent = 'nimcp_health_agent_macros.h' in initial_includes
    has_mesh_participant = 'nimcp_mesh_participant.h' in initial_includes
    has_mesh_adapter = 'nimcp_mesh_adapter.h' in initial_includes

    # Check which non-boilerplate includes are in the boilerplate region
    extra_includes = []
    for i in range(bp_start, health_agent_line):
        line = lines[i].strip()
        if '#include' in line:
            if any(h in line for h in [
                'nimcp_health_agent_macros.h',
                'nimcp_mesh_participant.h',
                'nimcp_mesh_adapter.h',
                'nimcp_logging.h',
                '<stddef.h>',
                'nimcp_bridge_boilerplate.h',
            ]):
                continue
            include_name = re.search(r'#include\s+[<"]([^>"]+)[>"]', line)
            if include_name:
                inc = include_name.group(1)
                if inc not in initial_includes:
                    extra_includes.append(line)

    # Build the replacement block
    replacement_lines = []

    # Add separator if the original had one
    if lines[bp_start].strip().startswith('//='):
        replacement_lines.append('//=============================================================================')

    # Add necessary includes
    if not has_stddef:
        replacement_lines.append('#include <stddef.h>')
    if not has_logging:
        replacement_lines.append('#include "utils/logging/nimcp_logging.h"')
    if not has_health_agent:
        replacement_lines.append('#include "utils/fault_tolerance/nimcp_health_agent_macros.h"')

    replacement_lines.append('#include "utils/bridge/nimcp_bridge_boilerplate.h"')

    if not has_mesh_participant:
        replacement_lines.append('#include "mesh/nimcp_mesh_participant.h"')
    if not has_mesh_adapter:
        replacement_lines.append('#include "mesh/nimcp_mesh_adapter.h"')

    for inc in extra_includes:
        replacement_lines.append(inc)

    replacement_lines.append('')
    replacement_lines.append(f'BRIDGE_BOILERPLATE_MESH_ONLY({module_name}, {category})')

    # Combine: lines before boilerplate + replacement + lines after boilerplate
    new_lines = lines[:bp_start] + replacement_lines + lines[bp_end + 1:]

    new_content = '\n'.join(new_lines)

    with open(filepath, 'w') as f:
        f.write(new_content)

    lines_removed = (bp_end - bp_start + 1) - len(replacement_lines)
    print(f"  OK: {os.path.basename(filepath)} - module={module_name}, category={category}, removed ~{lines_removed} lines")
    return True


def main():
    files = find_files()
    print(f"Found {len(files)} files with mesh registration boilerplate to process")
    print(f"(Already processed files with BRIDGE_BOILERPLATE_MESH_ONLY are skipped)")
    print()

    success = 0
    skip = 0
    fail = 0

    for filepath in files:
        try:
            if process_file(filepath):
                success += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  FAIL: {filepath}: {e}")
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\nDone: {success} processed, {skip} skipped, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
