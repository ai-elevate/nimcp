#!/usr/bin/env python3
"""Replace NIMCP_CHECK_THROW / NIMCP_CHECK_THROW_IMMUNE / NIMCP_CHECK_THROW_MSG
with NIMCP_FEP_CHECK_THROW in FEP bridge files.

FEP bridge functions return 0/-1, not NIMCP_ERROR_* codes.
NIMCP_CHECK_THROW does `return (code)` which returns the NIMCP error code.
NIMCP_FEP_CHECK_THROW does `return -1` instead.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MACROS_TO_REPLACE = [
    'NIMCP_CHECK_THROW_IMMUNE',  # longest first to avoid partial match
    'NIMCP_CHECK_THROW_MSG',
    'NIMCP_CHECK_THROW',
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
    """Replace CHECK_THROW macros with FEP_CHECK_THROW."""
    with open(filepath, 'r') as f:
        content = f.read()

    replacements = 0
    for macro in MACROS_TO_REPLACE:
        # Only replace the macro name, not the arguments
        pattern = r'\b' + re.escape(macro) + r'\b'
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, 'NIMCP_FEP_CHECK_THROW', content)
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
    print(f"\n{action} {total_replacements} macro calls across {total_files_changed} files")

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
