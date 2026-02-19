#!/usr/bin/env python3
"""Fix FEP bridge return values: NIMCP_ERROR_* -> -1, NIMCP_OK/NIMCP_SUCCESS -> 0.

FEP bridge functions return int (0 for success, -1 for errors).
Many files incorrectly return NIMCP_ERROR_* codes or NIMCP_OK/NIMCP_SUCCESS.
This script fixes them all.
"""

import os
import re
import sys

# Root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Error code patterns to replace with -1
ERROR_CODES = [
    'NIMCP_ERROR_NULL_POINTER',
    'NIMCP_ERROR_INVALID_STATE',
    'NIMCP_ERROR_INVALID_PARAM',
    'NIMCP_ERROR_OPERATION_FAILED',
    'NIMCP_ERROR_NO_MEMORY',
    'NIMCP_ERROR_NULL_ARG',
    'NIMCP_ERROR_MEMORY',
    'NIMCP_ERROR_INVALID_ARGUMENT',
    'NIMCP_ERROR_NOT_INITIALIZED',
    'NIMCP_ERROR_ALREADY_INITIALIZED',
    'NIMCP_ERROR_TIMEOUT',
    'NIMCP_ERROR_BUSY',
    'NIMCP_ERROR_UNKNOWN',
]

# Success codes to replace with 0
SUCCESS_CODES = [
    'NIMCP_OK',
    'NIMCP_SUCCESS',
]

def find_fep_bridge_files(root):
    """Find all *_fep_bridge*.c files."""
    matches = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, 'src')):
        for f in filenames:
            if 'fep_bridge' in f and f.endswith('.c'):
                matches.append(os.path.join(dirpath, f))
    return sorted(matches)


def fix_file(filepath, dry_run=False):
    """Fix return values in a single FEP bridge file."""
    with open(filepath, 'r') as f:
        original = f.read()

    content = original
    replacements = 0

    # Replace error code returns
    for code in ERROR_CODES:
        pattern = r'\breturn\s+' + re.escape(code) + r'\s*;'
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, 'return -1;', content)
            replacements += len(matches)

    # Replace success code returns
    for code in SUCCESS_CODES:
        pattern = r'\breturn\s+' + re.escape(code) + r'\s*;'
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, 'return 0;', content)
            replacements += len(matches)

    if replacements > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements


def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    files = find_fep_bridge_files(ROOT)
    print(f"Found {len(files)} FEP bridge files")

    total_replacements = 0
    total_files_changed = 0

    for filepath in files:
        relpath = os.path.relpath(filepath, ROOT)
        count = fix_file(filepath, dry_run=dry_run)
        if count > 0:
            total_replacements += count
            total_files_changed += 1
            if verbose:
                print(f"  {relpath}: {count} replacements")

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_replacements} return values across {total_files_changed} files")

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
