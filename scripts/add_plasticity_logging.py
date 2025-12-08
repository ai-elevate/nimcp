#!/usr/bin/env python3
"""
Add comprehensive logging to all Plasticity module files.

This script:
1. Adds #include "utils/logging/nimcp_logging.h" if not present
2. Adds #define LOG_MODULE "plasticity_<name>" after includes
3. Adds logging statements to:
   - Create/init functions
   - Destroy/cleanup functions
   - Error paths
   - Key operations
   - Validation failures
"""

import os
import re
import sys

# Logging statement templates
TEMPLATES = {
    'create': 'LOG_INFO("Creating {module}");',
    'create_success': 'LOG_INFO("{module} created successfully");',
    'destroy': 'LOG_INFO("Destroying {module}");',
    'error_alloc': 'LOG_ERROR("Failed to allocate memory for {module}");',
    'error_null': 'LOG_ERROR("NULL parameter in {func}");',
    'error_invalid': 'LOG_ERROR("Invalid parameter in {func}: {detail}");',
    'warn_invalid': 'LOG_WARN("Invalid parameter: {detail}");',
    'debug_update': 'LOG_DEBUG("{operation}: {detail}");',
}

def get_module_name(filepath):
    """Extract module name from file path."""
    basename = os.path.basename(filepath)
    name = basename.replace('nimcp_', '').replace('.c', '')
    return name

def has_logging_include(content):
    """Check if file already includes logging header."""
    return 'nimcp_logging.h' in content

def has_log_module_define(content):
    """Check if file already has LOG_MODULE defined."""
    return re.search(r'#define\s+LOG_MODULE\s+"', content) is not None

def add_logging_include(content):
    """Add logging include if not present."""
    if has_logging_include(content):
        return content

    # Find the last #include line
    lines = content.split('\n')
    last_include_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            last_include_idx = i

    if last_include_idx >= 0:
        lines.insert(last_include_idx + 1, '#include "utils/logging/nimcp_logging.h"')
        return '\n'.join(lines)

    return content

def add_log_module_define(content, module_name):
    """Add LOG_MODULE define after includes."""
    if has_log_module_define(content):
        return content

    lines = content.split('\n')

    # Find position after last include and before first function/constant
    insert_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            insert_idx = i + 1
        elif insert_idx > 0 and line.strip() and not line.strip().startswith('#'):
            break

    if insert_idx > 0:
        # Add blank line and define
        lines.insert(insert_idx, '')
        lines.insert(insert_idx + 1, f'#define LOG_MODULE "plasticity_{module_name}"')
        lines.insert(insert_idx + 2, '')
        return '\n'.join(lines)

    return content

def add_logging_to_function(content, func_pattern, log_statements):
    """Add logging statements to a function."""
    # This is a simplified approach - we'll add logging at key points
    lines = content.split('\n')
    modified = False
    new_lines = []

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Add logging after function start
        if func_pattern in line and '{' in line:
            # Add LOG_INFO for create/init functions
            if 'create' in func_pattern or 'init' in func_pattern:
                if i + 1 < len(lines) and 'LOG_' not in lines[i + 1]:
                    new_lines.append('    ' + log_statements.get('entry', ''))
                    modified = True

        # Add logging before returns
        stripped = line.strip()
        if stripped.startswith('return NULL') or stripped == 'return false;':
            # Check if previous line already has LOG_ERROR
            if i > 0 and 'LOG_ERROR' not in lines[i - 1]:
                # Add error logging before return
                indent = len(line) - len(line.lstrip())
                new_lines.insert(-1, ' ' * indent + log_statements.get('error', ''))
                modified = True

    return '\n'.join(new_lines) if modified else content

def process_file(filepath):
    """Process a single file to add logging."""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    module_name = get_module_name(filepath)

    # Step 1: Add include if needed
    content = add_logging_include(content)

    # Step 2: Add LOG_MODULE define if needed
    content = add_log_module_define(content, module_name)

    # Step 3: Add logging statements to key functions
    # We'll do manual pattern matching for common patterns

    # Find create/init functions
    create_pattern = re.compile(r'(\w+_(?:create|init))\s*\([^)]*\)\s*{', re.MULTILINE)
    for match in create_pattern.finditer(content):
        func_name = match.group(1)
        # Check if next line already has logging
        start_pos = match.end()
        next_lines = content[start_pos:start_pos+200]
        if 'LOG_INFO' not in next_lines.split('\n')[0]:
            # Add logging
            indent = '    '
            log_stmt = f'{indent}LOG_INFO("Creating {module_name} via {func_name}");'
            content = content[:start_pos] + '\n' + log_stmt + content[start_pos:]

    # Find destroy/cleanup functions
    destroy_pattern = re.compile(r'(\w+_(?:destroy|cleanup|free))\s*\([^)]*\)\s*{', re.MULTILINE)
    for match in destroy_pattern.finditer(content):
        func_name = match.group(1)
        start_pos = match.end()
        next_lines = content[start_pos:start_pos+200]
        if 'LOG_INFO' not in next_lines.split('\n')[0]:
            indent = '    '
            log_stmt = f'{indent}LOG_INFO("Destroying {module_name} via {func_name}");'
            content = content[:start_pos] + '\n' + log_stmt + content[start_pos:]

    # Add error logging before "return NULL" in create functions
    lines = content.split('\n')
    new_lines = []
    in_create_func = False
    func_name = ''

    for i, line in enumerate(lines):
        # Detect if we're in a create function
        if '_create(' in line or '_init(' in line:
            match = re.search(r'(\w+_(?:create|init))\s*\(', line)
            if match:
                in_create_func = True
                func_name = match.group(1)

        # Add error logging before return NULL
        if in_create_func and 'return NULL' in line:
            # Check if previous line has LOG_ERROR
            if i > 0 and 'LOG_ERROR' not in lines[i-1]:
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + f'LOG_ERROR("Failed in {func_name}");')

        # Detect end of function
        if in_create_func and line.strip() == '}' and i > 0:
            in_create_func = False

        new_lines.append(line)

    content = '\n'.join(new_lines)

    # Count changes
    if content != original_content:
        # Count new LOG_ statements
        original_log_count = original_content.count('LOG_')
        new_log_count = content.count('LOG_')
        added_count = new_log_count - original_log_count

        # Write back
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ Added {added_count} logging statements to {os.path.basename(filepath)}")
        return added_count
    else:
        print(f"  - No changes needed for {os.path.basename(filepath)}")
        return 0

def main():
    plasticity_dir = '/home/bbrelin/nimcp/src/plasticity'

    if not os.path.exists(plasticity_dir):
        print(f"Error: Directory {plasticity_dir} not found")
        return 1

    # Find all .c files
    c_files = []
    for root, dirs, files in os.walk(plasticity_dir):
        for file in files:
            if file.endswith('.c'):
                c_files.append(os.path.join(root, file))

    print(f"Found {len(c_files)} C files in plasticity module\n")

    total_added = 0
    results = []

    for filepath in sorted(c_files):
        added = process_file(filepath)
        total_added += added
        if added > 0:
            results.append((os.path.basename(filepath), added))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(c_files)}")
    print(f"Total files modified: {len(results)}")
    print(f"Total logging statements added: {total_added}")
    print(f"\nFiles modified:")
    for filename, count in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {filename:40s} {count:3d} statements")

    return 0

if __name__ == '__main__':
    sys.exit(main())
