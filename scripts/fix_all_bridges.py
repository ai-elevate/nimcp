#!/usr/bin/env python3
"""
Comprehensive bridge_base_t refactoring script.

Fixes:
1. Headers: Add bridge_base.h include and bridge_base_t base to structs
2. Sources: Replace direct ->mutex/->bio_async_enabled/->bio_ctx with ->base.*
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def add_bridge_base_include(content):
    """Add bridge_base.h include if missing."""
    if 'utils/bridge/nimcp_bridge_base.h' in content:
        return content, False

    # Find a good insertion point (after other includes)
    match = re.search(r'(#include\s*[<"][^>\n"]+[>"][^\n]*\n)', content)
    if match:
        insert_pos = match.end()
        return content[:insert_pos] + '#include "utils/bridge/nimcp_bridge_base.h"\n' + content[insert_pos:], True

    return content, False

def fix_struct_in_content(content):
    """Add bridge_base_t base to the main bridge struct if missing."""
    if 'bridge_base_t base' in content:
        return content, False

    # Find typedef struct { ... } XXX_bridge_t; pattern
    pattern = r'(typedef\s+struct\s*\{\s*)([^}]+)(\}\s*\w+_bridge_t\s*;)'

    def replacer(match):
        prefix = match.group(1)
        body = match.group(2)
        suffix = match.group(3)

        # Check if this struct already has bridge_base_t
        if 'bridge_base_t base' in body:
            return match.group(0)

        # Remove old mutex/bio fields that will be in bridge_base_t
        fields_to_remove = [
            r'\s*nimcp_mutex_t\s*\*?\s*mutex\s*;[^\n]*\n?',
            r'\s*void\s*\*\s*mutex\s*;[^\n]*\n?',
            r'\s*bio_module_context_t\s+bio_ctx\s*;[^\n]*\n?',
            r'\s*bool\s+bio_async_enabled\s*;[^\n]*\n?',
            r'/\*\s*Bio-async integration\s*\*/\s*\n?',
            r'/\*\s*Thread safety\s*\*/\s*\n?',
        ]
        for field_pat in fields_to_remove:
            body = re.sub(field_pat, '', body)

        # Add bridge_base_t base as first member
        body = body.lstrip('\n')
        new_body = '\n    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */\n    ' + body.lstrip()

        return prefix + new_body + suffix

    new_content, count = re.subn(pattern, replacer, content, flags=re.DOTALL)
    return new_content, count > 0

def fix_source_accesses(content):
    """Replace direct bridge-> accesses with bridge->base. for base fields."""
    modified = False

    # Only replace if we're not already using bridge->base.
    if 'bridge->mutex' in content or 'bridge->bio_async_enabled' in content or 'bridge->bio_ctx' in content:
        content = re.sub(r'\bbridge->mutex\b', 'bridge->base.mutex', content)
        content = re.sub(r'\bbridge->bio_async_enabled\b', 'bridge->base.bio_async_enabled', content)
        content = re.sub(r'\bbridge->bio_ctx\b', 'bridge->base.bio_ctx', content)
        modified = True

    return content, modified

def process_file(filepath, is_header):
    """Process a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    modified = False

    # Check if this file has bridge types
    has_bridge_type = '_bridge_t' in content
    if not has_bridge_type:
        return False

    if is_header:
        # Add include
        new_content, incl_mod = add_bridge_base_include(content)
        if incl_mod:
            content = new_content
            modified = True

        # Fix struct
        new_content, struct_mod = fix_struct_in_content(content)
        if struct_mod:
            content = new_content
            modified = True
    else:
        # Add include
        new_content, incl_mod = add_bridge_base_include(content)
        if incl_mod:
            content = new_content
            modified = True

        # Fix accesses
        new_content, access_mod = fix_source_accesses(content)
        if access_mod:
            content = new_content
            modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("Comprehensive Bridge Refactoring")
    print("=" * 60)

    # Process headers
    header_count = 0
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if file.endswith('.h'):
                filepath = Path(root) / file
                if process_file(filepath, is_header=True):
                    print(f"Fixed header: {filepath.relative_to(PROJECT_ROOT)}")
                    header_count += 1

    print(f"\nFixed {header_count} headers")

    # Process sources
    source_count = 0
    src_dir = PROJECT_ROOT / 'src'
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.c'):
                filepath = Path(root) / file
                if process_file(filepath, is_header=False):
                    print(f"Fixed source: {filepath.relative_to(PROJECT_ROOT)}")
                    source_count += 1

    print(f"Fixed {source_count} sources")
    print(f"\nTotal: {header_count + source_count} files")

if __name__ == '__main__':
    main()
