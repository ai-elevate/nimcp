#!/usr/bin/env python3
"""
Fix remaining mutex references after bridge_base refactoring.

This script:
1. Removes `bool mutex_initialized;` field declarations from headers
2. Updates `bridge->mutex_initialized` to `(bridge->base.mutex != NULL)` in sources
3. Updates `bridge->mutex` to `bridge->base.mutex` where still direct
4. Updates macro definitions that use direct mutex references
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_header_file(filepath):
    """Remove mutex_initialized field from header file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Remove mutex_initialized field declarations
    # Match: bool mutex_initialized; with optional comment
    content = re.sub(
        r'\n\s*bool\s+mutex_initialized\s*;\s*(?://[^\n]*)?',
        '',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def fix_source_file(filepath):
    """Fix mutex references in source file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Pattern 1: Replace direct mutex access in macros
    # LOCK/UNLOCK macros that use bridge->mutex
    content = re.sub(
        r'\(bridge\)->mutex(?![_a-zA-Z])',
        '(bridge)->base.mutex',
        content
    )
    content = re.sub(
        r'bridge->mutex(?![_a-zA-Z])',
        'bridge->base.mutex',
        content
    )
    content = re.sub(
        r'mutable_bridge->mutex(?![_a-zA-Z])',
        'mutable_bridge->base.mutex',
        content
    )

    # Pattern 2: Replace mutex_initialized = true/false with mutex allocation
    # bridge->mutex_initialized = true; => (keep, but we need to set something)
    # Actually these are usually after nimcp_mutex_init, so we can remove them
    content = re.sub(
        r'\n\s*bridge->mutex_initialized\s*=\s*true\s*;',
        '',
        content
    )
    content = re.sub(
        r'\n\s*bridge->mutex_initialized\s*=\s*false\s*;',
        '',
        content
    )

    # Pattern 3: Replace mutex_initialized checks with mutex != NULL
    content = re.sub(
        r'bridge->mutex_initialized(?!\s*=)',
        '(bridge->base.mutex != NULL)',
        content
    )

    # Pattern 4: Handle &bridge->base.mutex for inline mutex (not pointer)
    # nimcp_mutex_lock(&bridge->base.mutex) should be nimcp_mutex_lock(bridge->base.mutex)
    # because base.mutex is already a pointer
    content = re.sub(
        r'nimcp_mutex_lock\(&bridge->base\.mutex\)',
        'nimcp_mutex_lock(bridge->base.mutex)',
        content
    )
    content = re.sub(
        r'nimcp_mutex_unlock\(&bridge->base\.mutex\)',
        'nimcp_mutex_unlock(bridge->base.mutex)',
        content
    )
    content = re.sub(
        r'nimcp_mutex_lock\(&mutable_bridge->base\.mutex\)',
        'nimcp_mutex_lock(mutable_bridge->base.mutex)',
        content
    )
    content = re.sub(
        r'nimcp_mutex_unlock\(&mutable_bridge->base\.mutex\)',
        'nimcp_mutex_unlock(mutable_bridge->base.mutex)',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("=" * 60)
    print("Fixing remaining mutex references")
    print("=" * 60)

    header_count = 0
    source_count = 0

    # Fix headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if fix_header_file(filepath):
                print(f"Fixed header: {filepath.relative_to(PROJECT_ROOT)}")
                header_count += 1

    # Fix sources
    src_dir = PROJECT_ROOT / 'src'
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue
            filepath = Path(root) / file
            if fix_source_file(filepath):
                print(f"Fixed source: {filepath.relative_to(PROJECT_ROOT)}")
                source_count += 1

    print(f"\nFixed {header_count} headers and {source_count} sources")

if __name__ == '__main__':
    main()
