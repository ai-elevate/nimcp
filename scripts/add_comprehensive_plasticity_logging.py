#!/usr/bin/env python3
"""
Add comprehensive logging to all Plasticity module files.
This script adds detailed logging to:
- Create/init functions (entry + success)
- Destroy/cleanup functions
- All error paths (before return NULL/false)
- Update/apply functions (debug level)
- Validation failures (warnings)
- Key plasticity operations (weight updates, etc.)
"""

import os
import re
import sys

def add_comprehensive_logging(filepath):
    """Add comprehensive logging to a single file."""

    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    i = 0
    module_name = os.path.basename(filepath).replace('nimcp_', '').replace('.c', '')

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Pattern 1: Add success logging at end of create functions
        if ('_create(' in line or '_init(' in line) and '{' in line:
            # Extract function name
            match = re.search(r'(\w+_(?:create|init))\s*\(', line)
            if match:
                func_name = match.group(1)
                # Look for the return statement at end of function
                j = i + 1
                indent = None
                while j < len(lines) and j < i + 200:  # Look ahead max 200 lines
                    if 'return' in lines[j] and ';' in lines[j]:
                        # Check it's a success return (not NULL/false)
                        if 'return NULL' not in lines[j] and 'return false' not in lines[j]:
                            # Add LOG_DEBUG before return
                            if 'LOG_' not in lines[j-1]:
                                indent = len(lines[j]) - len(lines[j].lstrip())
                                log_line = ' ' * indent + f'LOG_DEBUG("{func_name} completed successfully");\n'
                                new_lines.append(log_line)
                                modified = True
                    if lines[j].strip() == '}' and (j - i) > 10:  # End of function
                        break
                    j += 1

        # Pattern 2: Add error logging before "return NULL"
        if 'return NULL' in line and 'LOG_ERROR' not in lines[max(0,i-2):i]:
            indent = len(line) - len(line.lstrip())
            # Try to determine context
            context_info = "allocation failed"
            if 'malloc' in lines[max(0,i-5):i][-1]:
                context_info = "memory allocation failed"
            elif 'config' in lines[max(0,i-5):i][-1]:
                context_info = "invalid configuration"

            new_lines.insert(-1, ' ' * indent + f'LOG_ERROR("Failed: {context_info}");\n')
            modified = True

        # Pattern 3: Add error logging before "return false"
        if line.strip() == 'return false;' and 'LOG_ERROR' not in lines[max(0,i-2):i]:
            # Check if this is in a validation function
            if i > 10:
                for j in range(max(0, i-20), i):
                    if '_validate(' in lines[j] or '_check(' in lines[j]:
                        indent = len(line) - len(line.lstrip())
                        new_lines.insert(-1, ' ' * indent + f'LOG_WARN("Validation failed");\n')
                        modified = True
                        break

        # Pattern 4: Add debug logging for weight updates
        if 'weight' in line.lower() and ('=' in line or '+=' in line or '*=' in line):
            # Check if this is an actual weight update (not a comparison)
            if '==' not in line and '!=' not in line and 'weight[' not in line:
                # Check if next few lines don't have LOG_DEBUG
                if i + 1 < len(lines) and 'LOG_DEBUG' not in ''.join(lines[i:min(i+3,len(lines))]):
                    indent = len(line) - len(line.lstrip())
                    # Extract weight variable name
                    match = re.search(r'(\w*weight\w*)\s*[=+\-*]=', line)
                    if match:
                        weight_var = match.group(1)
                        new_lines.append(' ' * indent + f'LOG_DEBUG("Weight updated: {weight_var} = %.4f", {weight_var});\n')
                        modified = True

        # Pattern 5: Add warning for parameter clamping
        if 'clamp' in line.lower() and '(' in line:
            if i + 1 < len(lines) and 'LOG_' not in ''.join(lines[i:min(i+2,len(lines))]):
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + f'LOG_DEBUG("Parameter clamped");\n')
                modified = True

        # Pattern 6: Add info logging in destroy functions
        if ('_destroy(' in line or '_cleanup(' in line or '_free(' in line) and '{' in line:
            match = re.search(r'(\w+_(?:destroy|cleanup|free))\s*\(', line)
            if match:
                func_name = match.group(1)
                # Check if next line has LOG_INFO
                if i + 1 < len(lines) and 'LOG_INFO' not in lines[i+1]:
                    # Get indent from next non-empty line
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        indent = len(lines[j]) - len(lines[j].lstrip())
                        new_lines.append(' ' * indent + f'LOG_DEBUG("{func_name} cleaning up resources");\n')
                        modified = True

        # Pattern 7: Add debug logging for plasticity operations
        if any(keyword in line.lower() for keyword in ['stdp', 'ltp', 'ltd', 'plasticity', 'bcm', 'eligibility']):
            if ('apply' in line or 'update' in line) and '(' in line:
                # Check if this is a function call or definition
                if i + 2 < len(lines) and 'LOG_' not in ''.join(lines[i:min(i+5,len(lines))]):
                    if '{' in line or (i+1 < len(lines) and '{' in lines[i+1]):
                        # This is a function definition
                        indent = '    '
                        match = re.search(r'(\w+)\s*\(', line)
                        if match:
                            func_name = match.group(1)
                            new_lines.append(indent + f'LOG_DEBUG("{func_name} executing plasticity update");\n')
                            modified = True

        i += 1

    if modified:
        # Write back
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

        # Count added statements
        original_count = ''.join(lines).count('LOG_')
        new_count = ''.join(new_lines).count('LOG_')
        added = new_count - original_count

        return added

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

    print(f"Adding comprehensive logging to {len(c_files)} plasticity files...\n")

    total_added = 0
    results = []

    for filepath in sorted(c_files):
        print(f"Processing {os.path.basename(filepath)}...", end=' ')
        added = add_comprehensive_logging(filepath)
        total_added += added

        if added > 0:
            print(f"✓ Added {added} logging statements")
            results.append((os.path.basename(filepath), added))
        else:
            print("- No additional logging needed")

    print(f"\n{'='*60}")
    print("COMPREHENSIVE LOGGING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(c_files)}")
    print(f"Total files modified: {len(results)}")
    print(f"Total logging statements added: {total_added}")

    if results:
        print(f"\nFiles modified:")
        for filename, count in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"  {filename:40s} +{count:3d} statements")

    return 0

if __name__ == '__main__':
    sys.exit(main())
