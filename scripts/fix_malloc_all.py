#!/usr/bin/env python3
"""
Automated NIMCP Memory Policy Enforcement - Complete Edition
Replaces ALL raw malloc/calloc/realloc/free with nimcp_* equivalents
across the entire source tree (src/ and include/).

Excludes:
  - include/utils/memory/nimcp_memory.h (the wrapper itself)
  - src/utils/memory/nimcp_memory.c (wrapper implementation)
  - include/utils/platform/ and src/utils/platform/ (platform layer)
  - venv/ directories
  - test/ directories
"""

import os
import re
import sys

# Patterns with negative lookbehind to only match standalone malloc/free/calloc/realloc
PATTERNS = [
    (re.compile(r'(?<![_a-zA-Z0-9])malloc\('), 'nimcp_malloc('),
    (re.compile(r'(?<![_a-zA-Z0-9])calloc\('), 'nimcp_calloc('),
    (re.compile(r'(?<![_a-zA-Z0-9])realloc\('), 'nimcp_realloc('),
    (re.compile(r'(?<![_a-zA-Z0-9])free\('), 'nimcp_free('),
]

# Include pattern to add
MEMORY_INCLUDE = '#include "utils/memory/nimcp_memory.h"'

# Paths to exclude
EXCLUDED_PATHS = [
    'include/utils/memory/nimcp_memory.h',
    'src/utils/memory/nimcp_memory.c',
    'include/utils/platform/',
    'src/utils/platform/',
    'venv/',
    'test/',
]


def should_exclude(filepath):
    for ex in EXCLUDED_PATHS:
        if ex in filepath:
            return True
    return False


def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return False, False

    original_content = content

    for pattern, replacement in PATTERNS:
        content = pattern.sub(replacement, content)

    if content == original_content:
        return False, False

    added_include = False
    if MEMORY_INCLUDE not in content and 'nimcp_memory.h' not in content:
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
        if insert_idx > 0:
            lines.insert(insert_idx, MEMORY_INCLUDE)
            content = '\n'.join(lines)
            added_include = True

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"  [ERROR] Cannot write {filepath}: {e}")
        return False, False

    return True, added_include


def main():
    print("=" * 79)
    print("NIMCP Memory Policy Enforcement - Complete Source Tree Conversion")
    print("=" * 79)

    root_dirs = ['src/', 'include/']
    fixed = 0
    includes_added = 0
    total_scanned = 0

    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in sorted(filenames):
                if not (fname.endswith('.c') or fname.endswith('.h')):
                    continue
                fpath = os.path.join(dirpath, fname)
                if should_exclude(fpath):
                    continue
                total_scanned += 1
                changed, inc_added = fix_file(fpath)
                if changed:
                    fixed += 1
                    status = "[FIXED]"
                    if inc_added:
                        status += " [+include]"
                        includes_added += 1
                    print(f"  {status} {fpath}")

    print()
    print("=" * 79)
    print(f"Summary:")
    print(f"  Scanned: {total_scanned} files")
    print(f"  Fixed:   {fixed} files")
    print(f"  Includes added: {includes_added}")
    print("=" * 79)
    return 0


if __name__ == '__main__':
    sys.exit(main())
