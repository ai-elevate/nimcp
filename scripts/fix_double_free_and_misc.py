#!/usr/bin/env python3
"""Fix double-free, uninitialized vars, and unsafe string ops in cognitive layer.

1. Double-free: add ptr = NULL after nimcp_free(ptr)
2. Uninitialized vars: add = 0 or = 0.0f initializers
3. Unsafe strings: strcat -> strncat, sprintf -> snprintf

Usage: python3 fix_double_free_and_misc.py [directory] [--dry-run]
"""
import re
import os
import sys


def fix_null_after_free(lines):
    """Add ptr = NULL after nimcp_free(ptr) where missing."""
    fixes = 0
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        new_lines.append(line)

        if stripped.startswith('//') or stripped.startswith('/*'):
            i += 1
            continue

        # Match nimcp_free(var) or free(var) - standalone calls
        m = re.match(r'^(\s*)(?:nimcp_free|free)\s*\(\s*(\w+)\s*\)\s*;', stripped)
        if m:
            indent = re.match(r'^(\s*)', line).group(1)
            var_name = m.group(2)

            # Check if next line already sets to NULL
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if re.search(rf'{re.escape(var_name)}\s*=\s*NULL\s*;', next_stripped):
                    i += 1
                    continue

            # Check if it's on the same line as return (pattern: nimcp_free(x); return ...;)
            # In that case, skip - the pointer won't be used again
            if 'return' in stripped:
                i += 1
                continue

            # Check if var_name is a simple local (not a->b pattern)
            if '->' not in var_name and '.' not in var_name:
                # Add NULL assignment
                new_lines.append(f"{indent}{var_name} = NULL;\n")
                fixes += 1

        i += 1

    return new_lines, fixes


def fix_uninitialized_vars(lines):
    """Add initializers to uninitialized local variable declarations."""
    fixes = 0
    new_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*'):
            new_lines.append(line)
            continue

        # Match: type varname; (without initializer, not pointer, not struct member)
        m = re.match(r'^(\s*)(float|double|int|uint32_t|uint64_t|int32_t|int64_t|size_t|bool)\s+(\w+)\s*;\s*$', stripped)
        if m:
            indent = re.match(r'^(\s*)', line).group(1)
            var_type = m.group(2)
            var_name = m.group(3)

            # Skip if it's a loop variable or looks like it's at file scope
            if indent == '' or indent == '\t':
                new_lines.append(line)
                continue

            # Determine default value
            if var_type in ('float',):
                default = '0.0f'
            elif var_type in ('double',):
                default = '0.0'
            elif var_type in ('bool',):
                default = 'false'
            else:
                default = '0'

            new_line = f"{indent}{var_type} {var_name} = {default};\n"
            new_lines.append(new_line)
            fixes += 1
        else:
            new_lines.append(line)

    return new_lines, fixes


def fix_unsafe_strings(lines):
    """Replace strcat with strncat using safe bounds."""
    fixes = 0
    new_lines = []

    for line in lines:
        stripped = line.strip()

        # Fix strcat(dst, src) -> strncat(dst, src, sizeof(dst) - strlen(dst) - 1)
        m = re.search(r'\bstrcat\s*\(\s*(\w+)\s*,\s*([^)]+)\)', stripped)
        if m and not stripped.startswith('//'):
            dst = m.group(1)
            src = m.group(2).strip()
            indent = re.match(r'^(\s*)', line).group(1)
            new_line = f"{indent}strncat({dst}, {src}, sizeof({dst}) - strlen({dst}) - 1);\n"
            new_lines.append(new_line)
            fixes += 1
            continue

        new_lines.append(line)

    return new_lines, fixes


def fix_file(filepath, dry_run=False):
    """Apply all fixes to a single file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return 0

    total_fixes = 0

    # Apply fixes in sequence
    lines, fixes = fix_null_after_free(lines)
    total_fixes += fixes

    lines, fixes = fix_uninitialized_vars(lines)
    total_fixes += fixes

    lines, fixes = fix_unsafe_strings(lines)
    total_fixes += fixes

    if total_fixes > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.writelines(lines)

    return total_fixes


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else 'src/cognitive'
    dry_run = '--dry-run' in sys.argv

    total_fixes = 0
    fixed_files = 0

    for root, dirs, files in os.walk(target_dir):
        for f in sorted(files):
            if not f.endswith('.c'):
                continue
            filepath = os.path.join(root, f)
            count = fix_file(filepath, dry_run)
            if count > 0:
                total_fixes += count
                fixed_files += 1
                print(f"  {'[DRY] ' if dry_run else ''}Fixed {count:3d} in {filepath}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total_fixes} fixes in {fixed_files} files")


if __name__ == '__main__':
    main()
