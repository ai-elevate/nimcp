#!/usr/bin/env python3
"""
Fix typedef struct patterns that are missing bridge_base_t.
Handles: typedef struct X { ... } X;
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_typedef_struct(filepath):
    """Add bridge_base_t base to typedef struct if missing."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if has bridge_base.h include
    if 'bridge_base.h' not in content:
        return False

    # Check if already has bridge_base_t base
    if 'bridge_base_t base' in content:
        return False

    # Check if has typedef struct pattern with _bridge_t suffix
    if not re.search(r'typedef\s+struct\s+\w+_bridge_t\s*\{', content):
        return False

    # Find typedef struct X_bridge_t { ... } X_bridge_t;
    # Insert bridge_base_t base after the opening brace
    pattern = r'(typedef\s+struct\s+\w+_bridge_t\s*\{)(\s*\n)(\s*)'

    def add_base_member(match):
        typedef_open = match.group(1)
        ws = match.group(2)
        indent = match.group(3)
        return f"{typedef_open}{ws}{indent}bridge_base_t base;                  /**< MUST be first: base bridge infrastructure */\n\n{indent}"

    new_content, count = re.subn(pattern, add_base_member, content)

    if count > 0 and new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True

    return False

def main():
    print("=" * 60)
    print("Fixing typedef struct bridges")
    print("=" * 60)

    fixed_count = 0

    # Process headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if fix_typedef_struct(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} headers")

if __name__ == '__main__':
    main()
