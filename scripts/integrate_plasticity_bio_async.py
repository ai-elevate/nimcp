#!/usr/bin/env python3
"""
Add bio-async integration to all Plasticity module files.

This script systematically adds bio-async integration to all .c files
in the src/plasticity/ directory.
"""

import os
import re
from pathlib import Path

# Map filenames to their specific bio module IDs
MODULE_ID_MAP = {
    'nimcp_stdp.c': 'BIO_MODULE_STDP',
    'nimcp_stp.c': 'BIO_MODULE_STP',
    'nimcp_homeostatic.c': 'BIO_MODULE_HOMEOSTATIC',
    'nimcp_bcm.c': 'BIO_MODULE_BCM',
    'nimcp_dendritic.c': 'BIO_MODULE_DENDRITIC',
    'nimcp_adaptive.c': 'BIO_MODULE_ADAPTIVE',
    'nimcp_attention.c': 'BIO_MODULE_ATTENTION_PLASTICITY',
    'nimcp_predictive_coding.c': 'BIO_MODULE_PREDICTIVE_CODING',
    'nimcp_neuromodulators.c': 'BIO_MODULE_NEUROMODULATOR',
    'nimcp_pink_noise.c': 'BIO_MODULE_PINK_NOISE',
    'nimcp_eligibility_trace.c': 'BIO_MODULE_ELIGIBILITY_TRACE',
    'nimcp_spatial_neuromod.c': 'BIO_MODULE_NEUROMODULATOR_SPATIAL',
    'nimcp_neuromod_pink_noise.c': 'BIO_MODULE_NEUROMODULATOR_PINK_NOISE',
    'nimcp_metabolic_pathways.c': 'BIO_MODULE_NEUROMODULATOR_METABOLIC',
    'nimcp_receptor_subtypes.c': 'BIO_MODULE_NEUROMODULATOR_RECEPTOR',
    'nimcp_phasic_tonic.c': 'BIO_MODULE_NEUROMODULATOR_PHASIC_TONIC',
    'nimcp_vesicle_packaging.c': 'BIO_MODULE_NEUROMODULATOR_VESICLE',
}

def get_module_id(filename):
    """Get the BIO_MODULE_* constant for a given filename."""
    basename = os.path.basename(filename)
    return MODULE_ID_MAP.get(basename, 'BIO_MODULE_PLASTICITY')

def has_bio_async_includes(content):
    """Check if file already has bio-async includes."""
    return 'nimcp_bio_async.h' in content and 'nimcp_bio_messages.h' in content

def add_bio_async_includes(content):
    """Add bio-async includes to the file."""
    if has_bio_async_includes(content):
        return content, False

    # Find the last #include statement
    lines = content.split('\n')
    last_include_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            last_include_idx = i

    if last_include_idx == -1:
        # No includes found, add after file header comment
        for i, line in enumerate(lines):
            if '*/' in line:
                last_include_idx = i
                break

    # Insert bio-async includes
    insert_lines = [
        '#include "async/nimcp_bio_async.h"',
        '#include "async/nimcp_bio_messages.h"'
    ]

    lines[last_include_idx + 1:last_include_idx + 1] = insert_lines

    return '\n'.join(lines), True

def find_main_struct(content, filename):
    """Find the main struct definition in the file."""
    # Common patterns for plasticity structures
    patterns = [
        r'typedef\s+struct\s+(\w+)\s*{',  # typedef struct name {
        r'struct\s+(\w+)\s*{',             # struct name {
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            struct_name = match.group(1)
            # Skip forward declarations
            if 'typedef' in match.group(0) or '{' in content[match.end():match.end()+100]:
                return struct_name, match.start()

    return None, -1

def add_bio_fields_to_struct(content, struct_name):
    """Add bio-async fields to the struct."""
    if 'bio_async_context_t' in content and 'bio_async_enabled' in content:
        return content, False

    # Find the struct definition
    pattern = rf'(typedef\s+struct\s+{struct_name}\s*{{|struct\s+{struct_name}\s*{{)'
    match = re.search(pattern, content)

    if not match:
        return content, False

    # Find the closing brace
    start = match.end()
    brace_count = 1
    i = start
    while i < len(content) and brace_count > 0:
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count != 0:
        return content, False

    # Insert before the closing brace
    insert_pos = i - 1

    # Find the last field to insert after it
    last_field_pos = content.rfind(';', start, insert_pos)
    if last_field_pos == -1:
        insert_pos = start
    else:
        insert_pos = last_field_pos + 1

    bio_fields = '''

    /* Bio-async integration */
    bio_async_context_t* bio_ctx;
    bool bio_async_enabled;'''

    content = content[:insert_pos] + bio_fields + content[insert_pos:]

    return content, True

def add_registration_to_init(content, module_id):
    """Add bio-async registration to initialization function."""
    # Find init/create functions
    init_patterns = [
        r'(\w+)\s+(\w+_(?:init|create|new)(?:_\w+)?)\s*\([^)]*\)\s*{',
    ]

    modified = False
    for pattern in init_patterns:
        matches = list(re.finditer(pattern, content))
        for match in matches:
            func_name = match.group(2)
            func_start = match.end()

            # Find the function body
            brace_count = 1
            i = func_start
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1

            func_body = content[func_start:i-1]

            # Check if already has bio-async registration
            if 'bio_router_register_module' in func_body:
                continue

            # Find where to insert (after struct initialization, before return)
            # Look for patterns like "result->", "synapse->", "*obj = "
            insert_patterns = [
                r'(\w+)->(\w+)\s*=',  # result->field =
                r'memset\s*\([^)]+\)',  # memset(...)
            ]

            insert_pos = -1
            for ip in insert_patterns:
                last_match = None
                for m in re.finditer(ip, func_body):
                    last_match = m
                if last_match:
                    # Find the end of the statement (semicolon)
                    semi_pos = func_body.find(';', last_match.end())
                    if semi_pos != -1:
                        insert_pos = func_start + semi_pos + 1
                        break

            if insert_pos == -1:
                # Default: insert at the end of function before closing brace
                insert_pos = i - 2

            # Determine the result variable name
            result_var = 'result'
            if 'synapse->' in func_body:
                result_var = 'synapse'
            elif 'system->' in func_body:
                result_var = 'system'
            elif '*' in match.group(0):
                # Try to extract variable name from function signature
                var_match = re.search(r'\*(\w+)\s*=', func_body)
                if var_match:
                    result_var = var_match.group(1)

            registration_code = f'''

    /* Bio-async integration */
    {result_var}->bio_ctx = NULL;
    {result_var}->bio_async_enabled = false;
    bio_async_context_t* ctx = bio_router_get_global_context();
    if (ctx) {{
        {result_var}->bio_ctx = ctx;
        {result_var}->bio_async_enabled = bio_router_register_module(ctx, {module_id}, "{func_name}");
    }}'''

            content = content[:insert_pos] + registration_code + content[insert_pos:]
            modified = True
            break

    return content, modified

def add_unregistration_to_destroy(content):
    """Add bio-async unregistration to destroy/cleanup function."""
    # Find destroy/cleanup/free functions
    destroy_patterns = [
        r'void\s+(\w+_(?:destroy|cleanup|free|shutdown)(?:_\w+)?)\s*\([^)]*\)\s*{',
    ]

    modified = False
    for pattern in destroy_patterns:
        matches = list(re.finditer(pattern, content))
        for match in matches:
            func_start = match.end()

            # Find the function body
            brace_count = 1
            i = func_start
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1

            func_body = content[func_start:i-1]

            # Check if already has unregistration
            if 'bio_router_unregister_module' in func_body:
                continue

            # Determine the instance variable name
            instance_var = 'instance'
            if 'synapse->' in func_body:
                instance_var = 'synapse'
            elif 'system->' in func_body:
                instance_var = 'system'

            # Insert at the beginning of function
            insert_pos = func_start + 1

            unregistration_code = f'''
    /* Bio-async cleanup */
    if ({instance_var} && {instance_var}->bio_async_enabled && {instance_var}->bio_ctx) {{
        bio_router_unregister_module({instance_var}->bio_ctx, BIO_MODULE_PLASTICITY);
    }}
    '''

            content = content[:insert_pos] + unregistration_code + content[insert_pos:]
            modified = True
            break

    return content, modified

def add_event_broadcasting(content, module_id):
    """Add event broadcasting for key plasticity events."""
    # Find key plasticity functions (weight update, learning, etc.)
    event_functions = [
        r'float\s+(\w*(?:weight|stdp|update|learn|apply)\w*)\s*\([^)]*\)\s*{',
    ]

    modified = False
    for pattern in event_functions:
        matches = list(re.finditer(pattern, content))
        for match in matches[:3]:  # Limit to first 3 functions
            func_name = match.group(1)
            func_start = match.end()

            # Find the function body
            brace_count = 1
            i = func_start
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1

            func_body = content[func_start:i-1]

            # Check if already has broadcasting
            if 'bio_router_broadcast' in func_body:
                continue

            # Find return statement to insert before it
            return_pos = func_body.rfind('return')
            if return_pos == -1:
                continue

            insert_pos = func_start + return_pos

            # Determine instance variable
            instance_var = 'synapse'
            if 'system->' in func_body:
                instance_var = 'system'
            elif 'state->' in func_body:
                instance_var = 'state'

            broadcast_code = f'''

    /* Broadcast plasticity event */
    if ({instance_var} && {instance_var}->bio_async_enabled && {instance_var}->bio_ctx) {{
        bio_router_broadcast({instance_var}->bio_ctx, {module_id}, BIO_MSG_PLASTICITY_UPDATE, NULL, 0);
    }}
    '''

            content = content[:insert_pos] + broadcast_code + content[insert_pos:]
            modified = True
            break

    return content, modified

def process_file(filepath):
    """Process a single C file to add bio-async integration."""
    print(f"\nProcessing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # Get module ID for this file
    module_id = get_module_id(filepath)

    # Step 1: Add includes
    content, modified = add_bio_async_includes(content)
    if modified:
        changes.append("Added bio-async includes")

    # Step 2: Find and modify main struct (if exists)
    struct_name, _ = find_main_struct(content, filepath)
    if struct_name:
        content, modified = add_bio_fields_to_struct(content, struct_name)
        if modified:
            changes.append(f"Added bio-async fields to struct {struct_name}")

    # Step 3: Add registration to init functions
    content, modified = add_registration_to_init(content, module_id)
    if modified:
        changes.append("Added bio-async registration to init function")

    # Step 4: Add unregistration to destroy functions
    content, modified = add_unregistration_to_destroy(content)
    if modified:
        changes.append("Added bio-async unregistration to destroy function")

    # Step 5: Add event broadcasting
    content, modified = add_event_broadcasting(content, module_id)
    if modified:
        changes.append("Added bio-async event broadcasting")

    # Write back if modified
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ Modified ({len(changes)} changes):")
        for change in changes:
            print(f"    - {change}")
        return True, changes
    else:
        print(f"  ⊘ No changes needed")
        return False, []

def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent
    plasticity_dir = base_dir / 'src' / 'plasticity'

    if not plasticity_dir.exists():
        print(f"Error: Directory not found: {plasticity_dir}")
        return

    # Find all .c files
    c_files = list(plasticity_dir.rglob('*.c'))

    print(f"Found {len(c_files)} C files in plasticity module")
    print("=" * 80)

    modified_files = []
    all_changes = {}

    for c_file in sorted(c_files):
        was_modified, changes = process_file(c_file)
        if was_modified:
            modified_files.append(c_file)
            all_changes[str(c_file.relative_to(base_dir))] = changes

    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Total files: {len(c_files)}")
    print(f"  Modified: {len(modified_files)}")
    print(f"  Unchanged: {len(c_files) - len(modified_files)}")

    if modified_files:
        print(f"\n  Modified files:")
        for f in modified_files:
            print(f"    - {f.relative_to(base_dir)}")

    # Write summary report
    report_file = base_dir / 'PLASTICITY_BIO_ASYNC_INTEGRATION_REPORT.md'
    with open(report_file, 'w') as f:
        f.write("# Plasticity Module Bio-Async Integration Report\n\n")
        f.write(f"**Date:** {os.popen('date').read().strip()}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total files processed:** {len(c_files)}\n")
        f.write(f"- **Files modified:** {len(modified_files)}\n")
        f.write(f"- **Files unchanged:** {len(c_files) - len(modified_files)}\n")
        f.write(f"- **Integration completion:** {len(modified_files) / len(c_files) * 100:.1f}%\n\n")

        f.write("## Changes by File\n\n")
        for filepath, changes in sorted(all_changes.items()):
            f.write(f"### `{filepath}`\n\n")
            for change in changes:
                f.write(f"- {change}\n")
            f.write("\n")

        f.write("## Module ID Mapping\n\n")
        f.write("| File | Module ID |\n")
        f.write("|------|----------|\n")
        for filename, module_id in sorted(MODULE_ID_MAP.items()):
            f.write(f"| `{filename}` | `{module_id}` |\n")

    print(f"\n  Report written to: {report_file.relative_to(base_dir)}")

if __name__ == '__main__':
    main()
