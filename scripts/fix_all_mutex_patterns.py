#!/usr/bin/env python3
"""
Fix all mutex patterns to work with pointer-based mutex in bridge_base_t.

The bridge_base_t has: nimcp_mutex_t* mutex;  (a pointer, not inline mutex)

So all &bridge->base.mutex patterns are WRONG and need to be fixed:
- pthread_mutex_init(&bridge->base.mutex, ...) => wrong
- nimcp_platform_mutex_init(&bridge->base.mutex, ...) => wrong
- nimcp_mutex_lock(&bridge->base.mutex) => wrong

All should use bridge->base.mutex directly (without &).
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_file(filepath):
    """Fix all &bridge->base.mutex patterns."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Check if file has &bridge->base.mutex pattern
    if '&bridge->base.mutex' not in content:
        return False

    # Replace all &bridge->base.mutex with bridge->base.mutex
    # But we need to be careful about casts like (pthread_mutex_t*)&bridge->base.mutex

    # Pattern 1: Cast before & - e.g., (pthread_mutex_t*)&bridge->base.mutex
    content = re.sub(
        r'\([^)]+\)\s*&bridge->base\.mutex',
        'bridge->base.mutex',
        content
    )

    # Pattern 2: Direct & usage - e.g., &bridge->base.mutex
    content = re.sub(
        r'&bridge->base\.mutex',
        'bridge->base.mutex',
        content
    )

    # Also handle mutable_bridge variant
    content = re.sub(
        r'\([^)]+\)\s*&mutable_bridge->base\.mutex',
        'mutable_bridge->base.mutex',
        content
    )
    content = re.sub(
        r'&mutable_bridge->base\.mutex',
        'mutable_bridge->base.mutex',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("=" * 60)
    print("Fixing all &bridge->base.mutex patterns")
    print("=" * 60)

    fixed_count = 0

    # Fix sources
    src_dir = PROJECT_ROOT / 'src'
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
