#!/usr/bin/env python3
"""Fix test expectations for FEP bridge return values.

After fixing FEP bridges to return 0/-1 (instead of NIMCP_ERROR_*/NIMCP_OK),
tests that check for specific error codes need to be updated.

This script finds patterns like:
  EXPECT_EQ(result, NIMCP_ERROR_NULL_POINTER)  -> EXPECT_EQ(result, -1)
  EXPECT_EQ(result, NIMCP_ERROR_INVALID_PARAM) -> EXPECT_EQ(result, -1)
  ASSERT_EQ(result, NIMCP_ERROR_NULL_POINTER)  -> ASSERT_EQ(result, -1)
  EXPECT_EQ(result, NIMCP_OK)                  -> EXPECT_EQ(result, 0)

Only in test files that test FEP bridge functions.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Error codes to replace with -1
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
    'NIMCP_ERROR_UNKNOWN',
]

# Success codes to replace with 0
SUCCESS_CODES = [
    'NIMCP_OK',
    'NIMCP_SUCCESS',
]


def find_fep_test_files(root):
    """Find test files that test FEP bridges."""
    matches = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, 'test')):
        for f in filenames:
            if f.endswith('.cpp') or f.endswith('.c'):
                filepath = os.path.join(dirpath, f)
                # Check if file references FEP bridge functions
                try:
                    with open(filepath, 'r') as fh:
                        content = fh.read()
                    if 'fep_bridge' in content.lower() or 'fep_bridge' in f.lower():
                        matches.append(filepath)
                except (IOError, UnicodeDecodeError):
                    pass
    return sorted(matches)


def fix_file(filepath, dry_run=False):
    """Fix EXPECT_EQ/ASSERT_EQ expectations for FEP bridge returns."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    replacements = 0

    # Only fix lines that contain FEP bridge function calls or results that
    # are clearly from FEP bridge operations.
    # Pattern: EXPECT_EQ(var, NIMCP_ERROR_*) or ASSERT_EQ(var, NIMCP_ERROR_*)
    # where var is a result from a *_fep_* function call

    lines = content.split('\n')
    new_lines = []
    in_fep_context = False

    for i, line in enumerate(lines):
        # Track if we're in a FEP bridge test context
        if 'fep_bridge' in line.lower() or 'fep_' in line.lower():
            in_fep_context = True

        # Reset context at test function boundaries
        if line.strip().startswith('TEST_F(') or line.strip().startswith('TEST('):
            in_fep_context = 'fep' in line.lower()

        modified = line
        if in_fep_context:
            for code in ERROR_CODES:
                # Match EXPECT_EQ(something, NIMCP_ERROR_*)
                pattern = r'(EXPECT_EQ|ASSERT_EQ|EXPECT_NE|ASSERT_NE)\(([^,]+),\s*' + re.escape(code) + r'\)'
                if re.search(pattern, modified):
                    replacement_val = '-1'
                    modified = re.sub(pattern, r'\1(\2, ' + replacement_val + ')', modified)
                    replacements += 1

                # Match EXPECT_EQ(NIMCP_ERROR_*, something)  (reversed args)
                pattern2 = r'(EXPECT_EQ|ASSERT_EQ)\(' + re.escape(code) + r',\s*([^)]+)\)'
                if re.search(pattern2, modified):
                    modified = re.sub(pattern2, r'\1(-1, \2)', modified)
                    replacements += 1

            for code in SUCCESS_CODES:
                pattern = r'(EXPECT_EQ|ASSERT_EQ)\(([^,]+),\s*' + re.escape(code) + r'\)'
                if re.search(pattern, modified):
                    modified = re.sub(pattern, r'\1(\2, 0)', modified)
                    replacements += 1

                pattern2 = r'(EXPECT_EQ|ASSERT_EQ)\(' + re.escape(code) + r',\s*([^)]+)\)'
                if re.search(pattern2, modified):
                    modified = re.sub(pattern2, r'\1(0, \2)', modified)
                    replacements += 1

        new_lines.append(modified)

    content = '\n'.join(new_lines)

    if replacements > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements


def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    files = find_fep_test_files(ROOT)
    print(f"Found {len(files)} FEP-related test files")

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
        elif verbose and count == 0:
            pass  # skip files with no changes

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_replacements} expectations across {total_files_changed} files")

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
