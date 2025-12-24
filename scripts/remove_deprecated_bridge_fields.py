#!/usr/bin/env python3
"""
Remove deprecated duplicate fields from bridge structs that now have bridge_base_t base.

These fields are deprecated because they already exist in bridge_base_t:
- bio_module_context_t bio_ctx
- bool bio_async_enabled
- nimcp_mutex_t* mutex / void* mutex
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Patterns for deprecated fields (with optional comments and whitespace)
DEPRECATED_PATTERNS = [
    # bio_ctx field with various comment styles
    r'\s*/\*\*?[^*]*[Bb]io[- ]?async[^*]*\*/\s*\n',  # Block comment about bio-async
    r'\s*bio_module_context_t\s+bio_ctx\s*;[^\n]*\n',

    # bio_async_enabled field
    r'\s*bool\s+bio_async_enabled\s*;[^\n]*\n',

    # mutex field (both nimcp_mutex_t* and void*)
    r'\s*/\*\*?[^*]*[Tt]hread\s*safety[^*]*\*/\s*\n',  # Block comment about thread safety
    r'\s*nimcp_mutex_t\s*\*\s*mutex\s*;[^\n]*\n',
    r'\s*void\s*\*\s*mutex\s*;[^\n]*\n',

    # Empty line comments that might be left over
    r'\s*/\*\s*Bio-async\s*(integration|context)\s*\*/\s*\n',
    r'\s*/\*\s*Thread\s*safety\s*\*/\s*\n',
]

def has_bridge_base(content):
    """Check if struct has bridge_base_t base as first member."""
    return 'bridge_base_t base' in content or 'bridge_base_t base;' in content

def has_deprecated_fields(content):
    """Check if file has deprecated duplicate fields."""
    return ('bio_module_context_t bio_ctx' in content or
            'bool bio_async_enabled' in content or
            ('nimcp_mutex_t* mutex' in content or 'nimcp_mutex_t *mutex' in content or
             'void* mutex' in content or 'void *mutex' in content))

def remove_deprecated_fields(filepath):
    """Remove deprecated fields from a header file."""
    # NEVER process bridge_base.h itself - it DEFINES these fields!
    if 'nimcp_bridge_base.h' in str(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Only process files that have bridge_base_t base
    if not has_bridge_base(content):
        return False

    # Only process files that have deprecated fields
    if not has_deprecated_fields(content):
        return False

    original = content

    # Remove deprecated field patterns
    for pattern in DEPRECATED_PATTERNS:
        content = re.sub(pattern, '', content)

    # Clean up multiple consecutive blank lines (reduce to max 2)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Clean up blank lines before closing brace in struct
    content = re.sub(r'\n\n+(\s*\})', r'\n\1', content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("Removing deprecated duplicate fields from bridge structs")
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
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nRemoved deprecated fields from {fixed_count} headers")

if __name__ == '__main__':
    main()
