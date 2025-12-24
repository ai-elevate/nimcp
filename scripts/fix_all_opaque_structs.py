#!/usr/bin/env python3
"""
Comprehensive fix for opaque struct definitions in source files.
Adds bridge_base_t base as first member after struct opening brace.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_file(filepath):
    """Add bridge_base_t base to struct if file uses bridge->base. but doesn't have it."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if this file uses bridge->base. pattern
    if 'bridge->base.' not in content:
        return False

    # Check if already has bridge_base_t base
    if 'bridge_base_t base' in content:
        return False

    # Find struct { and insert base member right after
    # Pattern matches: struct <name> { followed by any whitespace/newlines
    pattern = r'(struct\s+\w+\s*\{)(\s*\n)'

    def replacer(match):
        struct_open = match.group(1)
        ws = match.group(2)
        return f"{struct_open}\\n    bridge_base_t base;                      /**< MUST be first: base bridge infrastructure */{ws}"

    # Use simple string replacement approach
    lines = content.split('\n')
    new_lines = []
    struct_found = False
    base_inserted = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        # Check if this line opens a struct definition
        if not base_inserted and re.match(r'\s*struct\s+\w+\s*\{', line):
            struct_found = True
            # Insert base member as next line
            new_lines.append('    bridge_base_t base;                      /**< MUST be first: base bridge infrastructure */')
            base_inserted = True

    if base_inserted:
        new_content = '\n'.join(new_lines)
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True

    return False

def main():
    print("=" * 60)
    print("Comprehensive Opaque Struct Fix")
    print("=" * 60)

    src_dir = PROJECT_ROOT / 'src'
    fixed_count = 0

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue

            filepath = Path(root) / file
            if fix_file(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
