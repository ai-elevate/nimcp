#!/usr/bin/env python3
"""
Safely remove deprecated duplicate fields from bridge structs.

Only removes specific field declaration lines, NOT struct closing braces.
Only processes files that have bridge_base_t base (which means these fields
are now duplicates).
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Lines to remove (exact patterns)
DEPRECATED_LINE_PATTERNS = [
    # bio_module_context_t bio_ctx with various spacing/comments
    r'^\s*bio_module_context_t\s+bio_ctx\s*;.*$',
    # bool bio_async_enabled
    r'^\s*bool\s+bio_async_enabled\s*;.*$',
    # nimcp_mutex_t* mutex (but NOT in bridge_base.h)
    r'^\s*nimcp_mutex_t\s*\*\s*mutex\s*;.*$',
    # void* mutex
    r'^\s*void\s*\*\s*mutex\s*;.*$',
]

# Comment-only lines that precede the deprecated fields
COMMENT_PATTERNS = [
    r'^\s*/\*\*?\s*Bio-async\s*(integration|context|module)?\s*\*/\s*$',
    r'^\s*/\*\*?\s*Thread\s*safety\s*(mutex)?\s*\*/\s*$',
    r'^\s*/\*\s*Bio-async\s*(integration|context|module)?\s*\*/\s*$',
    r'^\s*/\*\s*Thread\s*safety\s*\*/\s*$',
]

def has_bridge_base(content):
    """Check if file has bridge_base_t base."""
    return 'bridge_base_t base' in content

def is_bridge_base_file(filepath):
    """Check if this is bridge_base.h itself."""
    return 'nimcp_bridge_base.h' in str(filepath)

def remove_deprecated_fields(filepath):
    """Remove deprecated field lines from a file."""
    # Never process bridge_base.h
    if is_bridge_base_file(filepath):
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    content = ''.join(lines)

    # Only process files with bridge_base_t base
    if not has_bridge_base(content):
        return False

    # Process line by line
    new_lines = []
    skip_next_blank = False
    i = 0
    removed_any = False

    while i < len(lines):
        line = lines[i]
        should_remove = False

        # Check if this is a deprecated field line
        for pattern in DEPRECATED_LINE_PATTERNS:
            if re.match(pattern, line):
                should_remove = True
                removed_any = True
                break

        # Check if this is a comment preceding deprecated field
        if not should_remove:
            for pattern in COMMENT_PATTERNS:
                if re.match(pattern, line):
                    # Check if next non-blank line is a deprecated field
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1
                    if j < len(lines):
                        for fld_pattern in DEPRECATED_LINE_PATTERNS:
                            if re.match(fld_pattern, lines[j]):
                                should_remove = True
                                break
                    break

        if should_remove:
            # Skip blank lines after removed content
            skip_next_blank = True
        else:
            # Check if this is a blank line after removed content
            if skip_next_blank and line.strip() == '':
                # Skip one blank line after removal
                skip_next_blank = False
            else:
                skip_next_blank = False
                new_lines.append(line)

        i += 1

    if removed_any:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        return True

    return False

def main():
    print("=" * 60)
    print("Removing deprecated duplicate fields (safe version)")
    print("=" * 60)

    fixed_count = 0

    # Process headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if remove_deprecated_fields(filepath):
                print(f"Fixed header: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    # Process sources
    src_dir = PROJECT_ROOT / 'src'
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue
            filepath = Path(root) / file
            if remove_deprecated_fields(filepath):
                print(f"Fixed source: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nRemoved deprecated fields from {fixed_count} files")

if __name__ == '__main__':
    main()
