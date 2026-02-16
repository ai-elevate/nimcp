#!/usr/bin/env python3
"""Replace magic number timeouts with named constants from nimcp_timing_constants.h.

Focuses on safe patterns where context clearly indicates a timeout:
- struct fields named *timeout*
- function args with timeout parameter names
- Variable declarations with timeout in name
- #define macros with TIMEOUT in name

Usage:
    python3 scripts/replace_magic_timeouts.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_timing_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_timing_constants.h', 'nimcp_constants.h',
}

# Timeout value -> constant mapping
TIMEOUT_MAP = {
    '5000':   'NIMCP_DEFAULT_TIMEOUT_MS',
    '100':    'NIMCP_FAST_TIMEOUT_MS',
    '500':    'NIMCP_SHORT_TIMEOUT_MS',
    '1000':   'NIMCP_MEDIUM_TIMEOUT_MS',
    '30000':  'NIMCP_LONG_TIMEOUT_MS',
    '300000': 'NIMCP_EXTENDED_TIMEOUT_MS',
    '10000':  'NIMCP_WATCHDOG_TIMEOUT_MS',
    '2000':   'NIMCP_RECOVERY_L2_TIMEOUT_MS',
    '50':     'NIMCP_FAST_HEARTBEAT_MS',
}

# Patterns where a number is clearly a timeout
# 1. Variable/field assignment: .timeout = N or timeout_ms = N
TIMEOUT_ASSIGN_RE = re.compile(
    r'(\.\s*\w*timeout\w*\s*=\s*)'  # .timeout_field =
    r'(\d+)'                          # the number
    r'(\s*[;,])',                      # ; or ,
    re.IGNORECASE
)

# 2. Variable declaration: int/uint32_t/etc timeout = N
TIMEOUT_DECL_RE = re.compile(
    r'((?:int|uint\d+_t|unsigned|long|size_t)\s+\w*timeout\w*\s*=\s*)'
    r'(\d+)'
    r'(\s*;)',
    re.IGNORECASE
)

# 3. #define XXXX_TIMEOUT N
TIMEOUT_DEFINE_RE = re.compile(
    r'(#define\s+\w*TIMEOUT\w*\s+)'
    r'(\d+)'
    r'(\s*(?://.*|/\*.*)?$)',
    re.IGNORECASE | re.MULTILINE
)

# 4. Function call with named timeout arg: func(..., timeout, N, ...)
# This is harder to detect safely, skip for now.


def has_timing_include(content):
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_timing_constants.h' in content)


def add_include(content):
    """Add timing constants include after the first contiguous block of includes."""
    lines = content.split('\n')
    in_block = False
    last_include_in_block = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include'):
            in_block = True
            last_include_in_block = i
        elif in_block and stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('#'):
            break

    if last_include_in_block == -1:
        return content

    lines.insert(last_include_in_block + 1, f'#include {INCLUDE}')
    return '\n'.join(lines)


def replace_timeout(m, group_num=2):
    """Replace a timeout number with the named constant if it matches."""
    value = m.group(group_num)
    if value in TIMEOUT_MAP:
        parts = list(m.groups())
        parts[group_num - 1] = TIMEOUT_MAP[value]
        return ''.join(parts)
    return m.group(0)


def process_file(filepath, dry_run=False):
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return 0

    with open(filepath) as f:
        content = f.read()

    # Skip files that already have timing includes (they were likely already converted)
    # Actually, we should still check these - they may have leftover magic numbers

    original = content
    replacements = [0]  # Use list for closure mutation

    def make_replacer(group_num=2):
        def replacer(m):
            value = m.group(group_num)
            if value in TIMEOUT_MAP:
                # Don't replace if it's already a constant name
                full = m.group(0)
                if 'NIMCP_' in full:
                    return full
                replacements[0] += 1
                parts = list(m.groups())
                parts[group_num - 1] = TIMEOUT_MAP[value]
                return ''.join(parts)
            return m.group(0)
        return replacer

    content = TIMEOUT_ASSIGN_RE.sub(make_replacer(2), content)
    content = TIMEOUT_DECL_RE.sub(make_replacer(2), content)
    content = TIMEOUT_DEFINE_RE.sub(make_replacer(2), content)

    if replacements[0] == 0:
        return 0

    # Add include if needed
    if not has_timing_include(content):
        content = add_include(content)

    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements[0]


def main():
    dry_run = '--dry-run' in sys.argv

    total_replacements = 0
    files_modified = 0
    files_checked = 0

    for root, dirs, files in os.walk(os.path.join(ROOT, 'src')):
        dirs[:] = [d for d in dirs if d not in ('build', '.git', '__pycache__')]
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            files_checked += 1
            count = process_file(filepath, dry_run)
            if count > 0:
                rel = os.path.relpath(filepath, ROOT)
                action = "WOULD FIX" if dry_run else "FIXED"
                print(f"  {action} ({count} replacements): {rel}")
                total_replacements += count
                files_modified += 1

    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {files_modified} files with {total_replacements} total replacements (checked {files_checked} files)")


if __name__ == '__main__':
    main()
