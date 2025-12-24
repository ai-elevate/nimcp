#!/usr/bin/env python3
"""
Comprehensive fix for all named struct bridge patterns.
Handles both headers and source files.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_named_struct_in_file(filepath):
    """Add bridge_base_t base to named struct if it's missing."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip files without bridge pattern
    if 'bridge->base.' not in content and 'bridge_base.h' not in content:
        return False

    # Check if already has bridge_base_t base in a struct
    if 'bridge_base_t base' in content:
        return False

    # Find ALL struct definitions with _bridge or _bridge_struct in name
    patterns = [
        r'(struct\s+\w+_bridge_struct\s*\{)(\s*\n)(\s*)',
        r'(struct\s+\w+_bridge\s*\{)(\s*\n)(\s*)',
    ]

    modified = False
    for pattern in patterns:
        def add_base_member(match):
            struct_open = match.group(1)
            ws = match.group(2)
            indent = match.group(3)
            return f"{struct_open}{ws}{indent}bridge_base_t base;               /**< MUST be first: base bridge infrastructure */\n\n{indent}"

        new_content, count = re.subn(pattern, add_base_member, content)
        if count > 0 and new_content != content:
            content = new_content
            modified = True
            break

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("Fixing ALL named struct bridges")
    print("=" * 60)

    fixed_count = 0

    # Process headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if fix_named_struct_in_file(filepath):
                print(f"Fixed header: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    # Process sources
    src_dir = PROJECT_ROOT / 'src'
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue
            filepath = Path(root) / file
            if fix_named_struct_in_file(filepath):
                print(f"Fixed source: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
