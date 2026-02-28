#!/usr/bin/env python3
"""Fix const-correctness in headers, v3.

Cross-reference source definitions (non-const, with mutex_lock) against header
declarations. Fix any header that still has const for a fixed source function.
"""
import re
import os
import sys
import glob


def get_source_signatures(src_dir):
    """Get dict of func_name -> type_name from source files that call mutex_lock."""
    skip_types = {
        'cognitive_event_data_t', 'brain_event_t', 'resource_metrics_t',
        'rcog_subtask_result_t', 'surprise_thalamic_signal_t',
        'collective_fep_config_t'
    }

    signatures = {}
    for root, dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                lines = f.read().split('\n')

            i = 0
            while i < len(lines):
                line = lines[i]
                # Match non-const function definitions with type_t* param
                # e.g., "int func_name(type_t* param) {"
                # Must NOT have const before type_t
                m = re.search(r'(\w+)\s*\((\w+_t)\s*\*\s*(\w+)', line)
                if m and 'const ' + m.group(2) + '*' not in line:
                    func_name = m.group(1)
                    type_name = m.group(2)

                    if type_name in skip_types or not type_name.endswith(
                            ('_bridge_t', '_system_t', '_engine_t', '_t')):
                        i += 1
                        continue

                    # Check for opening brace (function definition, not declaration)
                    sig = line
                    j = i
                    while '{' not in sig and j < min(i + 10, len(lines) - 1):
                        j += 1
                        sig += '\n' + lines[j]
                    if '{' not in sig:
                        i += 1
                        continue

                    # Find body end
                    brace_count = 0
                    for k in range(i, len(lines)):
                        brace_count += lines[k].count('{') - lines[k].count('}')
                        if brace_count == 0 and k > i:
                            break
                    body = '\n'.join(lines[i:k + 1])

                    if 'mutex_lock' in body:
                        signatures[func_name] = type_name
                i += 1

    return signatures


def fix_header_files(include_dir, signatures, base_dir):
    """Fix header files that still have const for fixed functions."""
    total = 0
    file_count = 0

    for root, dirs, files in os.walk(include_dir):
        for fname in sorted(files):
            if not fname.endswith('.h'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            original = content
            fixes = 0

            for func_name, type_name in signatures.items():
                # Pattern: func_name(const type_name* ...);
                # Could be on same line or split across lines
                pattern = func_name + r'\s*\('
                const_target = 'const ' + type_name + '*'

                # Find all occurrences of func_name in the file
                for match in re.finditer(pattern, content):
                    pos = match.start()
                    # Look in the next 200 chars for the const type_name*
                    snippet = content[pos:pos + 300]
                    if const_target in snippet and ';' in snippet:
                        # This is a declaration (has semicolon), fix it
                        # Calculate the exact position of const in the original content
                        const_pos = content.find(const_target, pos)
                        if const_pos != -1 and const_pos < pos + 300:
                            content = (content[:const_pos] +
                                      type_name + '*' +
                                      content[const_pos + len(const_target):])
                            fixes += 1

            if fixes > 0 and content != original:
                with open(fpath, 'w') as f:
                    f.write(content)
                file_count += 1
                total += fixes
                relpath = os.path.relpath(fpath, base_dir)
                print(f'{relpath}: {fixes} fixes')

    return total, file_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src', 'cognitive')
    include_dir = os.path.join(base_dir, 'include', 'cognitive')

    print("Phase 1: Collecting source function signatures...")
    signatures = get_source_signatures(src_dir)
    print(f"Found {len(signatures)} functions with mutex_lock and non-const params")

    # Print a sample
    for i, (fn, tn) in enumerate(sorted(signatures.items())[:10]):
        print(f"  {fn}({tn}*)")
    if len(signatures) > 10:
        print(f"  ... and {len(signatures) - 10} more")

    print("\nPhase 2: Fixing header declarations...")
    total, file_count = fix_header_files(include_dir, signatures, base_dir)

    print(f'\nTotal: {total} header fixes in {file_count} files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
