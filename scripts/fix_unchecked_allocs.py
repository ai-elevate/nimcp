#!/usr/bin/env python3
"""Fix unchecked nimcp_calloc/nimcp_malloc calls in cognitive layer.

Finds allocation calls where the return value is not checked for NULL,
and adds a NULL check with appropriate error handling.

Usage: python3 fix_unchecked_allocs.py [directory] [--dry-run]
"""
import re
import os
import sys


def fix_file(filepath, dry_run=False):
    """Fix unchecked allocations in a single file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return 0

    fixes = 0
    new_lines = list(lines)
    offset = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*'):
            continue

        # Match: var = nimcp_calloc(...) or var = nimcp_malloc(...)
        m = re.search(r'(\w+(?:\->[\w.]+)*)\s*=\s*(?:\([^)]*\)\s*)?(nimcp_calloc|nimcp_malloc|calloc|malloc)\s*\(', stripped)
        if not m:
            continue

        var_name = m.group(1)

        # Check next 4 lines for NULL check
        has_check = False
        for j in range(i + 1, min(i + 5, len(lines))):
            next_line = lines[j].strip()
            if re.search(r'if\s*\(\s*!' + re.escape(var_name), next_line):
                has_check = True
                break
            if re.search(r'if\s*\(\s*' + re.escape(var_name) + r'\s*==\s*NULL', next_line):
                has_check = True
                break
            if re.search(r'if\s*\(\s*NULL\s*==\s*' + re.escape(var_name), next_line):
                has_check = True
                break
            # goto/return/break counts as "handled"
            if re.match(r'\s*(goto|return|break)\b', next_line):
                has_check = True
                break
            # Another if on the same var
            if re.search(re.escape(var_name) + r'\s*\)', next_line) and 'if' in next_line:
                has_check = True
                break

        if has_check:
            continue

        # Need to add a NULL check
        indent = re.match(r'^(\s*)', line).group(1)

        # Determine what cleanup to do: if we're inside a function with goto cleanup, use that
        # Otherwise just return error
        # Look backwards for a 'cleanup:' label
        has_cleanup_label = False
        for j in range(i, max(0, i - 100), -1):
            if re.match(r'\s*(cleanup|error|fail|done)\s*:', lines[j]):
                has_cleanup_label = True
                label = re.match(r'\s*(\w+)\s*:', lines[j]).group(1)
                break

        if has_cleanup_label:
            check_lines = [
                f"{indent}if (!{var_name}) goto {label};\n"
            ]
        else:
            # Simple return
            check_lines = [
                f"{indent}if (!{var_name}) return -1;\n"
            ]

        # Insert after the allocation line
        insert_pos = i + 1 + offset
        for cl in check_lines:
            new_lines.insert(insert_pos, cl)
            insert_pos += 1
            offset += 1
        fixes += 1

    if fixes > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    return fixes


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
