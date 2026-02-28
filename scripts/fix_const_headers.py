#!/usr/bin/env python3
"""Fix const-correctness in header declarations to match source file fixes.

For each source file function that was changed from const type_t* to type_t*,
find and update the corresponding header declaration.
"""
import re
import os
import sys


def get_fixed_functions_from_source(src_dir):
    """Get list of (func_name, type_name) tuples from fixed source files."""
    skip_types = {
        'cognitive_event_data_t', 'brain_event_t', 'resource_metrics_t',
        'rcog_subtask_result_t', 'surprise_thalamic_signal_t',
        'collective_fep_config_t'
    }

    fixed = []
    for root, dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                # Find function definitions without const that have mutex_lock in body
                m = re.search(r'(\w+)\s*\(\s*(\w+_t)\s*\*\s*(\w+)', line)
                if m and 'const' not in line.split('(')[0]:  # No const before opening paren
                    func_name = m.group(1)
                    type_name = m.group(2)
                    if type_name in skip_types:
                        i += 1
                        continue

                    # Collect body
                    sig_start = i
                    sig = line
                    j = i
                    while '{' not in sig and j < min(i + 10, len(lines) - 1):
                        j += 1
                        sig += '\n' + lines[j]
                    if '{' not in sig:
                        i += 1
                        continue

                    brace_count = 0
                    for k in range(sig_start, len(lines)):
                        brace_count += lines[k].count('{') - lines[k].count('}')
                        if brace_count == 0 and k > sig_start:
                            break
                    body = '\n'.join(lines[sig_start:k + 1])

                    if 'mutex_lock' in body:
                        fixed.append((func_name, type_name))
                i += 1

    return fixed


def fix_headers(include_dir, fixed_funcs):
    """Fix header declarations to remove const for functions that lock mutexes."""
    total = 0
    file_count = 0

    func_set = set(fixed_funcs)

    for root, dirs, files in os.walk(include_dir):
        for fname in sorted(files):
            if not fname.endswith('.h'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            original = content
            lines = content.split('\n')
            fixes = 0

            for func_name, type_name in func_set:
                # Look for declaration pattern: func_name(const type_name*
                # Handle both single-line and multi-line declarations
                for li in range(len(lines)):
                    if func_name + '(' in lines[li] or func_name in lines[li]:
                        # Check this line and next few for const type_name*
                        for offset in range(min(5, len(lines) - li)):
                            target = 'const ' + type_name + '*'
                            if target in lines[li + offset]:
                                # Verify this is actually a declaration of the right function
                                context = '\n'.join(lines[max(0, li-1):li + offset + 2])
                                if func_name in context:
                                    lines[li + offset] = lines[li + offset].replace(
                                        target, type_name + '*', 1)
                                    fixes += 1
                                    break

            if fixes > 0:
                new_content = '\n'.join(lines)
                if new_content != original:
                    with open(fpath, 'w') as f:
                        f.write(new_content)
                    file_count += 1
                    total += fixes
                    relpath = os.path.relpath(fpath, os.path.dirname(include_dir))
                    print(f'{relpath}: {fixes} fixes')

    return total, file_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src', 'cognitive')
    include_dir = os.path.join(base_dir, 'include', 'cognitive')

    print("Phase 1: Collecting fixed functions from source files...")
    fixed_funcs = get_fixed_functions_from_source(src_dir)
    print(f"Found {len(fixed_funcs)} functions with mutex_lock and non-const params")

    print("\nPhase 2: Fixing header declarations...")
    total, file_count = fix_headers(include_dir, fixed_funcs)

    print(f'\nTotal: {total} header fixes in {file_count} files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
