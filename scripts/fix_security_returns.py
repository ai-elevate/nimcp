#!/usr/bin/env python3
"""
Fix bare 'return -1;' in security code with proper NIMCP error codes.
Analyzes the context above each 'return -1;' to choose the right error code.
Skips FEP bridge files and index/find/compare functions.
"""

import os
import re
import sys

BASE = "/home/bbrelin/nimcp/src/security"

# Files to skip entirely (FEP bridges use int 0/-1 convention intentionally)
SKIP_FILES = set()

# FEP bridge files - skip these
for root, dirs, files in os.walk(BASE):
    for f in files:
        if 'fep_bridge' in f:
            SKIP_FILES.add(os.path.join(root, f))

def classify_return(lines, return_line_idx):
    """Look at preceding lines to classify what error code to use."""
    # Look at the 5 lines before the return
    context = ""
    for i in range(max(0, return_line_idx - 5), return_line_idx):
        context += lines[i].lower()

    # NULL pointer checks
    if re.search(r'if\s*\(\s*!', context) and not re.search(r'size|count|len|cap|num', context):
        return "NIMCP_ERROR_NULL_POINTER"

    # Memory allocation failures
    if re.search(r'malloc|calloc|realloc|nimcp_malloc|nimcp_calloc', context):
        return "NIMCP_ERROR_NO_MEMORY"

    # Mutex/lock failures
    if re.search(r'mutex|lock', context):
        return "NIMCP_ERROR_MUTEX_INIT"

    # State checks (initialized, connected, etc.)
    if re.search(r'initializ|connected|active|enabled|state|ready|lockdown', context):
        return "NIMCP_ERROR_INVALID_STATE"

    # Parameter validation (size, count, etc.)
    if re.search(r'size.*==.*0|count.*==.*0|len.*==.*0|<.*0|>.*max|invalid|param', context):
        return "NIMCP_ERROR_INVALID_PARAM"

    # Default
    return "NIMCP_ERROR_OPERATION_FAILED"

def should_skip_function(lines, return_line_idx):
    """Check if this return -1 is in a function that should keep int return."""
    # Find the function signature by looking back for the function definition
    for i in range(return_line_idx, max(0, return_line_idx - 30), -1):
        line = lines[i]
        # Skip if in ssize_t, int32_t, or index-returning functions
        if re.search(r'ssize_t|int32_t|int16_t', line) and re.search(r'\w+\s*\(', line):
            return True
        # Skip find/search/compare functions
        if re.search(r'find_|search_|compare_|_cmp\(|_index\(|_slot\(', line):
            return True
        # Stop looking if we hit another function's closing brace
        if i < return_line_idx - 2 and line.strip() == '}':
            break
    return False

def fix_file(filepath):
    """Fix return -1 statements in a file."""
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    changes = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'return -1;':
            if should_skip_function(lines, i):
                continue
            error_code = classify_return(lines, i)
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}return {error_code};"
            changes += 1

    if changes > 0:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  [FIXED] {filepath} ({changes} changes)")

    return changes

def main():
    print("=" * 70)
    print("NIMCP Security Return Code Cleanup")
    print("=" * 70)

    total_changes = 0
    total_files = 0

    for root, dirs, files in os.walk(BASE):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            if filepath in SKIP_FILES:
                continue
            changes = fix_file(filepath)
            if changes > 0:
                total_files += 1
                total_changes += changes

    print(f"\nSummary: {total_changes} changes in {total_files} files")
    print("=" * 70)

if __name__ == '__main__':
    main()
