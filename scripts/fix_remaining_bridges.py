#!/usr/bin/env python3
"""
Fix remaining bridge files that are missing bridge_base_t base member.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_header_struct(content, struct_name):
    """Add bridge_base_t base to struct if missing."""
    if 'bridge_base_t base' in content:
        return content, False

    # Pattern: typedef struct { ... } name_t; or struct name { ... };
    patterns = [
        (rf'(typedef\s+struct\s*\{{\s*)([^}}]+)(\}}\s*{re.escape(struct_name)}\s*;)', True),
        (rf'(typedef\s+struct\s+\w+\s*\{{\s*)([^}}]+)(\}}\s*{re.escape(struct_name)}\s*;)', True),
        (rf'(struct\s+\w+\s*\{{\s*)([^}}]+)(\}}\s*;)', False),
    ]

    for pattern, is_typedef in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            prefix = match.group(1)
            body = match.group(2)
            suffix = match.group(3)

            # Remove old mutex/bio fields if present
            for field_pattern in [
                r'\s*nimcp_mutex_t\s*\*?\s*mutex\s*;[^\n]*\n',
                r'\s*void\s*\*\s*mutex\s*;[^\n]*\n',
                r'\s*bio_module_context_t\s+bio_ctx\s*;[^\n]*\n',
                r'\s*bool\s+bio_async_enabled\s*;[^\n]*\n',
            ]:
                body = re.sub(field_pattern, '\n', body)

            # Add bridge_base_t base as first member
            body = body.lstrip()
            new_body = '\n    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */\n    ' + body

            new_content = content[:match.start()] + prefix + new_body + suffix + content[match.end():]
            return new_content, True

    return content, False

def fix_source_accesses(content):
    """Replace direct ->mutex, ->bio_async_enabled, ->bio_ctx with ->base.*"""
    modified = False

    # Replace direct accesses with base. accesses
    replacements = [
        (r'bridge->mutex', 'bridge->base.mutex'),
        (r'bridge->bio_async_enabled', 'bridge->base.bio_async_enabled'),
        (r'bridge->bio_ctx', 'bridge->base.bio_ctx'),
    ]

    for old, new in replacements:
        if old in content and new not in content:
            # Only replace if we haven't already converted this file
            content = content.replace(old, new)
            modified = True

    return content, modified

def ensure_bridge_base_include(content):
    """Ensure bridge_base.h is included."""
    if 'utils/bridge/nimcp_bridge_base.h' in content:
        return content, False

    # Find first include
    match = re.search(r'#include\s*[<"]', content)
    if match:
        insert_pos = match.start()
        new_content = content[:insert_pos] + '#include "utils/bridge/nimcp_bridge_base.h"\n' + content[insert_pos:]
        return new_content, True

    return content, False

def process_headers():
    """Process all header files."""
    include_dir = PROJECT_ROOT / 'include'
    count = 0

    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue

            filepath = Path(root) / file
            with open(filepath, 'r') as f:
                content = f.read()

            # Skip if doesn't include bridge_base.h
            if 'bridge_base.h' not in content:
                continue

            # Skip if already has bridge_base_t base
            if 'bridge_base_t base' in content:
                continue

            # Find struct name from typedef
            match = re.search(r'typedef\s+struct\s*\{[^}]+\}\s*(\w+_bridge_t)\s*;', content, re.DOTALL)
            if not match:
                match = re.search(r'typedef\s+struct\s+\w+\s*\{[^}]+\}\s*(\w+_bridge_t)\s*;', content, re.DOTALL)

            if match:
                struct_name = match.group(1)
                new_content, modified = fix_header_struct(content, struct_name)
                if modified:
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print(f"Fixed header: {filepath.relative_to(PROJECT_ROOT)}")
                    count += 1

    return count

def process_sources():
    """Process all source files."""
    src_dir = PROJECT_ROOT / 'src'
    count = 0

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue

            filepath = Path(root) / file
            with open(filepath, 'r') as f:
                content = f.read()

            # Skip if no bridge pattern
            if 'bridge->' not in content:
                continue

            modified = False

            # Add bridge_base.h include if needed
            new_content, incl_modified = ensure_bridge_base_include(content)
            if incl_modified:
                content = new_content
                modified = True

            # Fix direct accesses
            new_content, access_modified = fix_source_accesses(content)
            if access_modified:
                content = new_content
                modified = True

            if modified:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"Fixed source: {filepath.relative_to(PROJECT_ROOT)}")
                count += 1

    return count

def main():
    print("=" * 60)
    print("Fixing remaining bridge files")
    print("=" * 60)

    header_count = process_headers()
    print(f"\nFixed {header_count} headers")

    source_count = process_sources()
    print(f"Fixed {source_count} sources")

    print(f"\nTotal: {header_count + source_count} files fixed")

if __name__ == '__main__':
    main()
