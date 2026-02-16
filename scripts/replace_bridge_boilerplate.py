#!/usr/bin/env python3
"""
Replace mesh registration boilerplate with BRIDGE_BOILERPLATE_MESH_ONLY macro.

This script processes .c files that have both NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
and mesh registration boilerplate (g_MODULE_mesh_id pattern), replacing them
with the new BRIDGE_BOILERPLATE_MESH_ONLY macro from nimcp_bridge_boilerplate.h.
"""

import os
import re
import sys

# Directories and their categories
DIR_CATEGORIES = {
    'src/dragonfly/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/embodiment/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/glial/': 'MESH_ADAPTER_CATEGORY_GLIAL',
    'src/integration/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/mesh/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/middleware/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/networking/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/physics/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/plasticity/': 'MESH_ADAPTER_CATEGORY_PLASTICITY',
    'src/portia/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/security/': 'MESH_ADAPTER_CATEGORY_SECURITY',
    'src/snn/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/superhuman/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/swarm/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/training/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/io/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/async/': 'MESH_ADAPTER_CATEGORY_SYSTEM',
    'src/lnn/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
    'src/chemistry/': 'MESH_ADAPTER_CATEGORY_COGNITIVE',
}

PROJECT_ROOT = '/home/bbrelin/nimcp'


def get_category_for_file(filepath):
    """Get the mesh adapter category for a file based on its directory."""
    rel = os.path.relpath(filepath, PROJECT_ROOT)
    for dir_prefix, category in DIR_CATEGORIES.items():
        if rel.startswith(dir_prefix):
            return category
    return None


def find_matching_brace(content, start_pos):
    """Find the matching closing brace from start_pos (which should point to '{')."""
    brace_count = 0
    for i in range(start_pos, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return i
    return -1


def process_file(filepath, dry_run=False):
    """Process a single file, replacing boilerplate with macro."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already processed
    if 'BRIDGE_BOILERPLATE_MESH_ONLY' in content or 'BRIDGE_BOILERPLATE(' in content:
        return False, "Already processed"

    # Extract module name from NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if not m:
        return False, "No NIMCP_DECLARE_HEALTH_AGENT_ATOMIC found"

    module_name = m.group(1)
    mesh_id_var = f'g_{module_name}_mesh_id'

    # Check if it has the mesh registration boilerplate
    if mesh_id_var not in content:
        return False, "No mesh registration boilerplate found"

    category = get_category_for_file(filepath)
    if not category:
        return False, "Unknown directory category"

    original = content

    # Step 1: Replace the health_agent_macros.h include with bridge_boilerplate.h
    content = content.replace(
        '#include "utils/fault_tolerance/nimcp_health_agent_macros.h"',
        '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    )

    # Step 2: Remove mesh/nimcp_mesh_participant.h and mesh/nimcp_mesh_adapter.h includes
    content = re.sub(r'#include "mesh/nimcp_mesh_participant\.h"\n', '', content)
    content = re.sub(r'#include "mesh/nimcp_mesh_adapter\.h"\n', '', content)

    # Step 3: Use manual parsing to find and remove the boilerplate block
    # Find the DECLARE line
    declare_match = re.search(
        r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(' + re.escape(module_name) + r'\)',
        content
    )
    if not declare_match:
        return False, "DECLARE line not found after include replacement"

    # Find the mesh registration section start
    mesh_section_start = content.find(f'static mesh_participant_id_t {mesh_id_var}')
    if mesh_section_start == -1:
        return False, "Could not find mesh registration section"

    # Find the end of mesh_unregister function
    unregister_start = content.find(f'void {module_name}_mesh_unregister(void)', mesh_section_start)
    if unregister_start == -1:
        # Try with space before void
        unregister_start = content.find(f'void {module_name}_mesh_unregister(void)', mesh_section_start)
    if unregister_start == -1:
        return False, "Could not find mesh_unregister function"

    # Find opening brace of unregister function
    open_brace = content.find('{', unregister_start)
    if open_brace == -1:
        return False, "Could not find opening brace of mesh_unregister"

    # Find matching closing brace
    close_brace = find_matching_brace(content, open_brace)
    if close_brace == -1:
        return False, "Could not find closing brace of mesh_unregister"

    unregister_end = close_brace + 1

    # Look backwards from mesh_section_start to find comment block
    search_start = max(0, mesh_section_start - 300)
    search_area = content[search_start:mesh_section_start]

    # Find the comment section header (//=== Mesh Participant Registration ===)
    comment_match = re.search(
        r'(//[=\-]+\s*\n//\s*Mesh Participant Registration\s*\n//[=\-]+\s*\n\s*)',
        search_area
    )
    if comment_match:
        actual_start = search_start + comment_match.start()
    else:
        actual_start = mesh_section_start

    # Consume trailing blank lines after the unregister function
    end_pos = unregister_end
    while end_pos < len(content) and content[end_pos] == '\n':
        end_pos += 1

    # Remove the mesh registration block
    content = content[:actual_start] + content[end_pos:]

    # Replace NIMCP_DECLARE_HEALTH_AGENT_ATOMIC with BRIDGE_BOILERPLATE_MESH_ONLY
    content = re.sub(
        r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(' + re.escape(module_name) + r'\)',
        f'BRIDGE_BOILERPLATE_MESH_ONLY({module_name}, {category})',
        content
    )

    if content == original:
        return False, "No changes made"

    if dry_run:
        return True, f"Would modify: {module_name} -> {category}"

    with open(filepath, 'w') as f:
        f.write(content)

    return True, f"Modified: {module_name} -> {category}"


def main():
    dry_run = '--dry-run' in sys.argv

    files_processed = 0
    files_skipped = 0
    files_failed = 0

    for dir_path in sorted(DIR_CATEGORIES.keys()):
        full_dir = os.path.join(PROJECT_ROOT, dir_path)
        if not os.path.isdir(full_dir):
            continue
        for root, dirs, files in os.walk(full_dir):
            dirs.sort()
            for fname in sorted(files):
                if not fname.endswith('.c'):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'r') as f:
                    content = f.read()

                # Only process files with mesh registration boilerplate
                if 'mesh_participant_register' not in content:
                    continue
                if 'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC' not in content:
                    continue

                m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
                if not m:
                    continue

                module_name = m.group(1)
                mesh_id_var = f'g_{module_name}_mesh_id'
                if mesh_id_var not in content:
                    continue

                success, msg = process_file(fpath, dry_run)
                if success:
                    files_processed += 1
                    print(f"  OK: {os.path.relpath(fpath, PROJECT_ROOT)} - {msg}")
                else:
                    if "Already processed" in msg:
                        files_skipped += 1
                        print(f"SKIP: {os.path.relpath(fpath, PROJECT_ROOT)} - {msg}")
                    else:
                        files_failed += 1
                        print(f"FAIL: {os.path.relpath(fpath, PROJECT_ROOT)} - {msg}")

    print(f"\nSummary: {files_processed} processed, {files_skipped} skipped, {files_failed} failed")


if __name__ == '__main__':
    main()
