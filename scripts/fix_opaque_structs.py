#!/usr/bin/env python3
"""
Fix opaque struct definitions in source files that are missing bridge_base_t base member.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_opaque_struct_in_file(filepath):
    """Add bridge_base_t base to opaque struct if it uses bridge->base. but doesn't have it."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if this file uses bridge->base. pattern
    if 'bridge->base.' not in content:
        return False

    # Check if already has bridge_base_t base in a struct
    if 'bridge_base_t base' in content:
        return False

    # Find struct definitions like: struct X_bridge { ... }; or struct X { ... };
    # Pattern for opaque struct in source file
    pattern = r'(struct\s+\w+\s*\{)(\s*)(\n)(\s*)(\w)'

    def add_base_member(match):
        struct_start = match.group(1)
        ws1 = match.group(2)
        newline = match.group(3)
        indent = match.group(4)
        first_char = match.group(5)

        # Add bridge_base_t base as first member
        return f"{struct_start}{ws1}{newline}{indent}bridge_base_t base;                      /**< MUST be first: base bridge infrastructure */{newline}{indent}{first_char}"

    new_content, count = re.subn(pattern, add_base_member, content)

    if count > 0:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True

    return False

def main():
    print("=" * 60)
    print("Fixing opaque structs in source files")
    print("=" * 60)

    src_dir = PROJECT_ROOT / 'src'
    fixed_count = 0

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue

            filepath = Path(root) / file
            if fix_opaque_struct_in_file(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
