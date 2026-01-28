#!/usr/bin/env python3
"""Fix silent returns in Phase 8 training functions across all src/cognitive/ files.

The Phase 8 upgrade script generated:
    if (!bridge) return -1;
    if (!instance) return -1;

These need NIMCP_THROW_TO_IMMUNE for proper exception handling.

Also fixes set_instance_health_agent to throw on null.
"""
import os
import re
import sys

SRC_ROOT = "/home/bbrelin/nimcp/src/cognitive"

# Patterns for Phase 8 training function null checks (single-line)
TRAINING_NULL_CHECK = re.compile(
    r'^(\s+)if\s*\((!(?:bridge|instance))\)\s*return\s+-1\s*;'
)

# Pattern to find function name from preceding lines
FUNC_DEF = re.compile(
    r'^(?:int|void)\s+(\w+(?:_training_begin|_training_end|_training_step|_set_instance_health_agent))\s*\('
)

# Pattern for set_instance_health_agent with simple guard
SET_INSTANCE_SIMPLE = re.compile(
    r'^(\s+)if\s*\(bridge\)\s*\{\s*bridge->health_agent\s*=\s*agent\s*;\s*\}'
)

# Check if file includes exception macros
HAS_EXCEPTION = re.compile(r'#include\s+"utils/exception/nimcp_exception_macros\.h"')

# Check if file has any NIMCP_THROW usage already (meaning it's set up for it)
HAS_THROW = re.compile(r'NIMCP_THROW')


def find_func_name(lines, idx):
    """Look backwards from idx to find the function name."""
    for i in range(idx, max(0, idx - 5), -1):
        m = FUNC_DEF.match(lines[i])
        if m:
            return m.group(1)
    return None


def needs_exception_include(content):
    """Check if file needs exception macros include added."""
    return not HAS_EXCEPTION.search(content)


def add_exception_include(lines):
    """Add exception macros include after the last #include line."""
    last_include = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            last_include = i
    if last_include >= 0:
        lines.insert(last_include + 1, '#include "utils/exception/nimcp_exception_macros.h"\n')
        return True
    return False


def process_file(filepath):
    """Fix Phase 8 silent returns in a single file."""
    with open(filepath) as f:
        content = f.read()

    # Only process files with Phase 8 functions
    if 'training_begin' not in content and 'set_instance_health_agent' not in content:
        return 0

    lines = content.split('\n')
    fixed = 0
    added_include = False

    # Check if we need to add exception include
    if needs_exception_include(content):
        # Only add if we'll actually fix something
        has_fixable = False
        for line in lines:
            if TRAINING_NULL_CHECK.match(line):
                has_fixable = True
                break
        if has_fixable:
            if add_exception_include(lines):
                added_include = True

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix training function null checks
        m = TRAINING_NULL_CHECK.match(line)
        if m:
            indent = m.group(1)
            var = m.group(2)  # e.g., "!bridge" or "!instance"
            func_name = find_func_name(lines, i)
            if func_name:
                replacement = [
                    f'{indent}if ({var}) {{',
                    f'{indent}    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,',
                    f'{indent}                          "{func_name}: NULL argument");',
                    f'{indent}    return -1;',
                    f'{indent}}}',
                ]
                lines[i] = '\n'.join(replacement)
                fixed += 1

        # Fix set_instance_health_agent simple guard
        m2 = SET_INSTANCE_SIMPLE.match(line)
        if m2:
            indent = m2.group(1)
            func_name = find_func_name(lines, i)
            if func_name:
                replacement = [
                    f'{indent}if (!bridge) {{',
                    f'{indent}    NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,',
                    f'{indent}                "{func_name}: NULL bridge");',
                    f'{indent}    return;',
                    f'{indent}}}',
                    f'{indent}bridge->health_agent = agent;',
                ]
                lines[i] = '\n'.join(replacement)
                fixed += 1

        i += 1

    if fixed > 0:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    return fixed


def main():
    dry_run = '--dry-run' in sys.argv

    total_fixed = 0
    total_files = 0
    files_modified = 0

    for root, dirs, files in os.walk(SRC_ROOT):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            total_files += 1

            if dry_run:
                with open(filepath) as f:
                    orig = f.read()
                fixed = process_file(filepath)
                if fixed > 0:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  WOULD FIX: {rel} ({fixed} fixes)")
                    total_fixed += fixed
                    files_modified += 1
                    # Restore
                    with open(filepath, 'w') as f:
                        f.write(orig)
            else:
                fixed = process_file(filepath)
                if fixed > 0:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  FIXED: {rel} ({fixed} fixes)")
                    total_fixed += fixed
                    files_modified += 1

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_fixed} silent returns across {files_modified}/{total_files} files")


if __name__ == '__main__':
    main()
