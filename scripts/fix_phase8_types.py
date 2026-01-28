#!/usr/bin/env python3
"""Fix Phase 8 type errors in set_instance_health_agent and training functions.

Two categories:
1. Pointer typedefs (typedef struct X* type_t) used with extra * -> pointer-to-pointer
2. Wrong type names extracted by the Phase 8 script

Strategy: For each .c file with Phase 8 functions, check if the type used matches
any type actually defined/used earlier in the file. If not, find the correct type
from the file's existing function signatures.
"""
import os
import re
import sys

SRC_ROOT = "/home/bbrelin/nimcp/src/cognitive"
INCLUDE_ROOT = "/home/bbrelin/nimcp/include"

# Find all pointer typedefs (typedef struct X* type_t)
def find_pointer_typedefs():
    """Return set of type names that are already pointers."""
    ptypes = set()
    for root, dirs, files in os.walk(INCLUDE_ROOT):
        for f in files:
            if not f.endswith('.h'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                for line in fh:
                    m = re.match(r'typedef\s+struct\s+\w+\s*\*\s*(\w+)\s*;', line)
                    if m:
                        ptypes.add(m.group(1))
    return ptypes


# Phase 8 function patterns
PHASE8_FUNCS = re.compile(
    r'^(?:void|int)\s+(\w+)_(set_instance_health_agent|training_begin|training_end|training_step)\s*\(\s*(\w+)\*?\s'
)

def find_correct_bridge_type(lines):
    """Find the bridge type used in existing functions (create/destroy/reset etc.)."""
    # Look for create/destroy function signatures with typed returns or params
    for line in lines:
        # Pattern: type_t* func_create(...) or void func_destroy(type_t* bridge)
        m = re.match(r'^(\w+_t)\*?\s+\w+_(?:create|destroy|reset|update|register)\s*\(', line)
        if m:
            return m.group(1)
        # Pattern: void func_destroy(type_t* bridge)
        m = re.match(r'^void\s+\w+_destroy\s*\(\s*(\w+_t)\*?\s', line)
        if m:
            return m.group(1)
        # Pattern: int func_reset(type_t* bridge)
        m = re.match(r'^int\s+\w+_reset\s*\(\s*(\w+_t)\*?\s', line)
        if m:
            return m.group(1)
    return None


def process_file(filepath, pointer_typedefs):
    """Fix Phase 8 type issues in a single file."""
    with open(filepath) as f:
        content = f.read()
        lines = content.split('\n')

    # Find what Phase 8 type is being used
    phase8_types = set()
    for line in lines:
        m = PHASE8_FUNCS.match(line)
        if m:
            phase8_types.add(m.group(3))

    if not phase8_types:
        return 0  # No Phase 8 functions

    # Find the correct bridge type from existing code
    correct_type = find_correct_bridge_type(lines)
    if not correct_type:
        return 0

    fixed = 0
    new_content = content

    for wrong_type in phase8_types:
        if wrong_type == correct_type:
            # Type is correct, but check if it's a pointer typedef used with *
            if correct_type in pointer_typedefs:
                # Replace type_t* with type_t (remove extra *)
                # But ONLY in Phase 8 function signatures, not throughout
                old_patterns = [
                    f'{correct_type}* bridge',
                    f'{correct_type} *bridge',
                ]
                for old in old_patterns:
                    if old in new_content:
                        # Check it's in a Phase 8 function context
                        new_content = fix_pointer_typedef(new_content, correct_type)
                        fixed += 1
                        break
        else:
            # Wrong type name - check if it's defined anywhere in the file
            if f'{wrong_type}' not in content.split('set_instance_health_agent')[0]:
                # Type doesn't exist before Phase 8 section - replace it
                if correct_type in pointer_typedefs:
                    # Pointer typedef: replace wrong_type* with correct_type (no *)
                    new_content = new_content.replace(f'{wrong_type}*', correct_type)
                    new_content = new_content.replace(f'{wrong_type} *', correct_type + ' ')
                else:
                    # Regular struct: replace wrong_type with correct_type
                    new_content = new_content.replace(f'{wrong_type}*', f'{correct_type}*')
                    new_content = new_content.replace(f'{wrong_type} ', f'{correct_type} ')
                fixed += 1

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)

    return fixed


def fix_pointer_typedef(content, ptype):
    """For pointer typedefs, remove extra * only in Phase 8 functions."""
    # These patterns only appear in Phase 8 sections
    phase8_markers = [
        'set_instance_health_agent',
        'training_begin',
        'training_end',
        'training_step',
    ]

    lines = content.split('\n')
    for i, line in enumerate(lines):
        for marker in phase8_markers:
            if marker in line and f'{ptype}*' in line:
                lines[i] = line.replace(f'{ptype}*', ptype)
            elif marker in line and f'{ptype} *' in line:
                lines[i] = line.replace(f'{ptype} *', ptype + ' ')

    return '\n'.join(lines)


def main():
    dry_run = '--dry-run' in sys.argv

    pointer_typedefs = find_pointer_typedefs()
    print(f"Found {len(pointer_typedefs)} pointer typedefs")

    total_fixed = 0
    for root, dirs, files in os.walk(SRC_ROOT):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)

            if dry_run:
                with open(filepath) as f:
                    orig = f.read()
                fixed = process_file(filepath, pointer_typedefs)
                if fixed > 0:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  WOULD FIX: {rel}")
                    total_fixed += fixed
                    # Restore original
                    with open(filepath, 'w') as f:
                        f.write(orig)
            else:
                fixed = process_file(filepath, pointer_typedefs)
                if fixed > 0:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  FIXED: {rel}")
                    total_fixed += fixed

    print(f"\n{'Would fix' if dry_run else 'Fixed'} {total_fixed} files")


if __name__ == '__main__':
    main()
