#!/usr/bin/env python3
"""Fix missing mutex unlock before return in cognitive layer.

Simple approach: For each return statement, look backwards for the nearest
mutex_lock or mutex_unlock call. If nearest is a lock (without intervening
unlock), insert nimcp_mutex_unlock() before the return.

Skips _unlocked functions and already-guarded returns.

Usage: python3 fix_missing_unlocks.py [directory] [--dry-run]
"""
import re
import os
import sys


def fix_file(filepath, dry_run=False):
    """Fix missing mutex unlocks in a single file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return 0

    content = ''.join(lines)

    # Quick check: does this file even have mutex_lock?
    if 'mutex_lock(' not in content:
        return 0

    fixes_needed = []  # (line_idx, mutex_var)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Only process return statements
        if not re.match(r'return\b', stripped):
            continue
        if stripped.startswith('//'):
            continue

        # Check if there's already an unlock in the 3 lines before
        has_nearby_unlock = False
        for j in range(max(0, i - 3), i):
            if 'mutex_unlock(' in lines[j] and not lines[j].strip().startswith('//'):
                has_nearby_unlock = True
                break
        if has_nearby_unlock:
            continue

        # Also skip if this return line itself has unlock
        if 'mutex_unlock(' in stripped:
            continue

        # Look backwards for nearest mutex_lock or mutex_unlock
        # Stop at function boundary indicators or max 200 lines back
        mutex_var = None
        in_comment_block = False

        for j in range(i - 1, max(0, i - 200), -1):
            jline = lines[j]
            jstripped = jline.strip()

            # Skip comments
            if jstripped.startswith('//'):
                continue
            if jstripped.endswith('*/'):
                in_comment_block = True
                continue
            if in_comment_block:
                if '/*' in jstripped:
                    in_comment_block = False
                continue
            if jstripped.startswith('/*'):
                continue

            # Function boundary: closing brace at column 0 (end of previous function)
            if jline.startswith('}'):
                break

            # Function boundary: function signature at column 0
            # Matches: return_type func_name(... { or return_type func_name(...) \n {
            if (re.match(r'^[a-zA-Z_]\w*[\s*]+\w+\s*\(', jline) and
                    not jstripped.startswith('if') and
                    not jstripped.startswith('for') and
                    not jstripped.startswith('while') and
                    not jstripped.startswith('switch') and
                    not jstripped.startswith('return') and
                    not jstripped.startswith('#')):
                # This looks like a function definition
                # Check if it's an _unlocked function
                if '_unlocked' in jline:
                    mutex_var = None
                    break
                # If we see a function sig, we've gone past any relevant lock
                break

            # Check for mutex_unlock — if found first, no fix needed
            if 'mutex_unlock(' in jstripped and not jstripped.startswith('//'):
                break

            # Check for mutex_lock — if found first, we need a fix
            m = re.search(r'mutex_lock\s*\(\s*([^)]+?)\s*\)', jstripped)
            if m and not jstripped.startswith('//'):
                mutex_var = m.group(1).strip()
                break

        if mutex_var:
            # Final check: is this inside an _unlocked function?
            # Look backwards for the function definition
            in_unlocked = False
            for j in range(i, max(0, i - 500), -1):
                jline = lines[j]
                # Function boundary at column 0
                if jline.startswith('}') and j < i - 1:
                    break
                # Function signature at column 0 with _unlocked
                if (re.match(r'^[a-zA-Z_]', jline) and '_unlocked' in jline and
                        ('(' in jline or (j + 1 < len(lines) and '(' in lines[j + 1]))):
                    in_unlocked = True
                    break

            if not in_unlocked:
                fixes_needed.append((i, mutex_var))

    if not fixes_needed:
        return 0

    # Deduplicate: avoid double-fixing the same line
    seen = set()
    unique_fixes = []
    for fix in fixes_needed:
        if fix[0] not in seen:
            seen.add(fix[0])
            unique_fixes.append(fix)
    fixes_needed = unique_fixes

    # Apply fixes in reverse order
    new_lines = list(lines)
    for return_idx, mutex_var in reversed(fixes_needed):
        indent = re.match(r'^(\s*)', new_lines[return_idx]).group(1)
        unlock_line = f"{indent}nimcp_mutex_unlock({mutex_var});\n"
        new_lines.insert(return_idx, unlock_line)

    if not dry_run:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    return len(fixes_needed)


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
