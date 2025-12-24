#!/usr/bin/env python3
"""
Fix named struct bridge definitions (struct X_bridge { ... };) in headers.
These use the forward declaration pattern with struct defined separately.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_named_struct_in_file(filepath):
    """Add bridge_base_t base to named struct if it's missing."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if file has bridge_base.h include
    if 'bridge_base.h' not in content:
        return False

    # Check if already has bridge_base_t base in a struct
    if 'bridge_base_t base' in content:
        return False

    # Check if this file has struct X_bridge { pattern
    if not re.search(r'struct\s+\w+_bridge\s*\{', content):
        return False

    # Find struct definitions: struct name_bridge { ... }
    # Insert bridge_base_t base after the opening brace
    pattern = r'(struct\s+\w+_bridge\s*\{)(\s*\n)(\s*)'

    def add_base_member(match):
        struct_open = match.group(1)
        ws = match.group(2)
        indent = match.group(3)
        return f"{struct_open}{ws}{indent}bridge_base_t base;               /**< MUST be first: base bridge infrastructure */\n\n{indent}"

    new_content, count = re.subn(pattern, add_base_member, content)

    if count > 0 and new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True

    return False

def main():
    print("=" * 60)
    print("Fixing named struct bridges in headers")
    print("=" * 60)

    include_dir = PROJECT_ROOT / 'include'
    fixed_count = 0

    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue

            filepath = Path(root) / file
            if fix_named_struct_in_file(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} headers")

if __name__ == '__main__':
    main()
