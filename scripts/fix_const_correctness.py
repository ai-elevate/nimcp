#!/usr/bin/env python3
"""Fix const-correctness violations in cognitive layer source files.

Finds functions with const type_t* parameters that call mutex_lock in their body,
and removes the const qualifier since locking a mutex modifies the struct (UB in C).
"""
import re
import os
import sys

def find_and_fix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # These types are input data params, not mutex-holding struct pointers
    skip_types = {
        'cognitive_event_data_t', 'brain_event_t', 'resource_metrics_t',
        'rcog_subtask_result_t', 'surprise_thalamic_signal_t',
        'collective_fep_config_t'
    }

    fixes = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'(\w+)\s*\(\s*const\s+(\w+_t)\s*\*\s*(\w+)', line)
        if m:
            type_name = m.group(2)
            if type_name in skip_types:
                i += 1
                continue

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
            body_end = sig_start
            for k in range(sig_start, len(lines)):
                brace_count += lines[k].count('{') - lines[k].count('}')
                if brace_count == 0 and k > sig_start:
                    body_end = k
                    break

            body = '\n'.join(lines[sig_start:body_end + 1])
            if 'mutex_lock' in body:
                for sl in range(sig_start, min(sig_start + 5, len(lines))):
                    target = 'const ' + type_name + '*'
                    if target in lines[sl]:
                        lines[sl] = lines[sl].replace(target, type_name + '*', 1)
                        fixes += 1
                        break
        i += 1

    if fixes > 0:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    return fixes


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cognitive_dir = os.path.join(base_dir, 'src', 'cognitive')

    total = 0
    file_count = 0

    for root, dirs, files in os.walk(cognitive_dir):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            fpath = os.path.join(root, fname)
            n = find_and_fix(fpath)
            if n > 0:
                file_count += 1
                total += n
                print(f'{os.path.relpath(fpath, base_dir)}: {n} fixes')

    print(f'\nTotal: {total} fixes in {file_count} files')
    return 0 if total > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
