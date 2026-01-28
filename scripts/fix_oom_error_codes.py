#!/usr/bin/env python3
"""Fix NIMCP_ERROR_NULL_POINTER used for allocation failures -> NIMCP_ERROR_NO_MEMORY.

Scans all .c files in src/cognitive/ for the pattern:
    ptr = nimcp_calloc/nimcp_malloc/calloc/malloc(...)
    if (!ptr) {
        ...
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");
        ...
    }

And replaces NIMCP_ERROR_NULL_POINTER with NIMCP_ERROR_NO_MEMORY, updating the message.
Handles both single-line and multi-line NIMCP_THROW calls.
"""
import os
import re
import sys

SRC_ROOT = "/home/bbrelin/nimcp/src/cognitive"

# Allocation function names
ALLOC_FUNCS = r'(?:nimcp_calloc|nimcp_malloc|nimcp_realloc|calloc|malloc|realloc)'

# Pattern to find NIMCP_THROW*(...NIMCP_ERROR_NULL_POINTER...) on a single line
THROW_PATTERN = re.compile(
    r'(NIMCP_THROW(?:_TO_IMMUNE)?)\s*\(\s*NIMCP_ERROR_NULL_POINTER\s*,'
)

# Pattern to detect allocation on a line
ALLOC_PATTERN = re.compile(
    rf'=\s*(?:\([^)]*\)\s*)?{ALLOC_FUNCS}\s*\('
)

# Message pattern on same line: "X is NULL"
MSG_SAME_LINE = re.compile(r'"(\w+)\s+is\s+NULL"')

# Message pattern on next line: "X is NULL"
MSG_NEXT_LINE = re.compile(r'^\s*"(\w+)\s+is\s+NULL"')

# How many lines to look back for an allocation before the throw
LOOKBACK = 12


def is_allocation_context(lines, throw_line_idx):
    """Check if there's an allocation within LOOKBACK lines before throw_line_idx."""
    start = max(0, throw_line_idx - LOOKBACK)
    for i in range(throw_line_idx - 1, start - 1, -1):
        line = lines[i]
        if ALLOC_PATTERN.search(line):
            return True
        # Stop if we hit a function definition
        if re.match(r'^[a-zA-Z].*\)\s*\{', line):
            break
    return False


def process_file(filepath):
    """Process a single .c file, returns (fixed_count, was_modified)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    fixed = 0
    modified = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if not THROW_PATTERN.search(line):
            i += 1
            continue

        # Check if this is in an allocation context
        if not is_allocation_context(lines, i):
            i += 1
            continue

        # Fix the error code on this line
        new_line = line.replace('NIMCP_ERROR_NULL_POINTER', 'NIMCP_ERROR_NO_MEMORY')

        # Try to fix message on same line
        msg_match = MSG_SAME_LINE.search(new_line)
        if msg_match:
            var_name = msg_match.group(1)
            new_line = new_line.replace(
                f'"{var_name} is NULL"',
                f'"Failed to allocate {var_name}"'
            )

        if new_line != line:
            lines[i] = new_line
            fixed += 1
            modified = True

        # Check if message is on the next line (multi-line throw)
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            msg_match_next = MSG_NEXT_LINE.search(next_line)
            if msg_match_next:
                var_name = msg_match_next.group(1)
                new_next = next_line.replace(
                    f'"{var_name} is NULL"',
                    f'"Failed to allocate {var_name}"'
                )
                if new_next != next_line:
                    lines[i + 1] = new_next
                    modified = True

        i += 1

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)

    return fixed, modified


def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

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
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                count = 0
                for idx, line in enumerate(lines):
                    if THROW_PATTERN.search(line) and is_allocation_context(lines, idx):
                        count += 1
                        if verbose:
                            rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                            print(f"  {rel}:{idx+1}: {line.strip()}")
                if count > 0:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  WOULD FIX: {rel} ({count} instances)")
                    total_fixed += count
                    files_modified += 1
            else:
                fixed, was_modified = process_file(filepath)
                if was_modified:
                    rel = os.path.relpath(filepath, '/home/bbrelin/nimcp')
                    print(f"  FIXED: {rel} ({fixed} instances)")
                    total_fixed += fixed
                    files_modified += 1

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_fixed} instances across {files_modified}/{total_files} files")


if __name__ == '__main__':
    main()
