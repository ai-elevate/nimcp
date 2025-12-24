#!/usr/bin/env python3
"""
Bridge Base Refactoring Script

Systematically updates all bridge files to use the bridge_base_t pattern.

Usage:
    python3 scripts/refactor_bridges.py [--dry-run]
"""

import os
import re
import sys
import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Patterns to match
BRIDGE_LOCK_PATTERN = re.compile(r'BRIDGE_LOCK\s*\(')
BIO_ASYNC_MACRO_PATTERN = re.compile(r'BRIDGE_DEFINE_BIO_ASYNC_FUNCS')

# Files to exclude (already properly refactored)
EXCLUDE_FILES = {
    'nimcp_working_memory_substrate_bridge',
    'nimcp_bridge_base',
}

def find_bridge_files():
    """Find all C files that use BRIDGE_LOCK or BRIDGE_DEFINE_BIO_ASYNC_FUNCS macros."""
    src_dir = PROJECT_ROOT / 'src'
    bridge_files = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.c'):
                continue

            # Skip excluded files
            base_name = file.replace('.c', '')
            if base_name in EXCLUDE_FILES:
                continue

            filepath = Path(root) / file
            with open(filepath, 'r') as f:
                content = f.read()

            if BRIDGE_LOCK_PATTERN.search(content) or BIO_ASYNC_MACRO_PATTERN.search(content):
                bridge_files.append(filepath)

    return bridge_files

def find_header_for_source(src_path):
    """Find the corresponding header file for a source file."""
    # Convert src path to include path
    relative = src_path.relative_to(PROJECT_ROOT / 'src')
    header_path = PROJECT_ROOT / 'include' / relative.with_suffix('.h')

    if header_path.exists():
        return header_path

    # Try alternate locations
    for include_root in [PROJECT_ROOT / 'include']:
        for root, dirs, files in os.walk(include_root):
            header_name = src_path.stem + '.h'
            if header_name in files:
                return Path(root) / header_name

    return None

def check_has_bridge_base(header_content):
    """Check if header struct already has bridge_base_t base."""
    return 'bridge_base_t base;' in header_content or 'bridge_base_t base' in header_content

def check_has_bridge_base_include(content):
    """Check if file includes bridge_base.h."""
    return 'utils/bridge/nimcp_bridge_base.h' in content

def add_bridge_base_include(content):
    """Add bridge_base.h include to file."""
    # Find first #include and add before it
    match = re.search(r'#include\s*[<"]', content)
    if match:
        insert_pos = match.start()
        return (content[:insert_pos] +
                '#include "utils/bridge/nimcp_bridge_base.h"\n' +
                content[insert_pos:])
    return content

def fix_bio_async_macro(content):
    """Fix BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE to use 2 arguments."""
    # Pattern: BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE(prefix, type, extra1, extra2)
    # Replace with: BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE(prefix, type)
    pattern = r'BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE\s*\(\s*(\w+)\s*,\s*(\w+)\s*,[^)]+\)'
    replacement = r'BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE(\1, \2)'
    return re.sub(pattern, replacement, content)

def extract_struct_name(header_content, src_name):
    """Extract the bridge struct name from header."""
    # Common patterns: typedef struct { ... } name_t; or struct name { ... };
    # Try to find typedef struct ending with _bridge_t
    patterns = [
        r'typedef\s+struct\s*\{[^}]+\}\s*(\w+_bridge_t)\s*;',
        r'typedef\s+struct\s+\w+\s*\{[^}]+\}\s*(\w+_bridge_t)\s*;',
        r'struct\s+(\w+_bridge)\s*\{',
    ]

    for pattern in patterns:
        match = re.search(pattern, header_content, re.DOTALL)
        if match:
            return match.group(1)

    # Fallback: guess from source file name
    base = src_name.replace('nimcp_', '').replace('.c', '')
    return f'{base}_t'

def update_struct_in_header(header_content, struct_name):
    """Update struct to use bridge_base_t base as first member."""
    # This is complex - we need to find the struct and modify it

    # Pattern to find struct definition
    struct_patterns = [
        # typedef struct { ... } name_t;
        (r'(typedef\s+struct\s*\{)([^}]+)(\}\s*' + re.escape(struct_name) + r'\s*;)',
         lambda m: update_struct_body(m, struct_name)),
        # typedef struct name { ... } name_t;
        (r'(typedef\s+struct\s+\w+\s*\{)([^}]+)(\}\s*' + re.escape(struct_name) + r'\s*;)',
         lambda m: update_struct_body(m, struct_name)),
        # struct name { ... };
        (r'(struct\s+\w+\s*\{)([^}]+)(\}\s*;)',
         lambda m: update_struct_body(m, struct_name)),
    ]

    for pattern, replacer in struct_patterns:
        match = re.search(pattern, header_content, re.DOTALL)
        if match:
            new_content = replacer(match)
            header_content = header_content[:match.start()] + new_content + header_content[match.end():]
            break

    return header_content

def update_struct_body(match, struct_name):
    """Update the struct body to use bridge_base_t."""
    prefix = match.group(1)
    body = match.group(2)
    suffix = match.group(3)

    # Check if already has bridge_base_t
    if 'bridge_base_t base' in body:
        return match.group(0)

    # Remove old fields that are now in bridge_base_t
    fields_to_remove = [
        r'\s*\w+\s*\*?\s*mutex\s*;',  # mutex field
        r'\s*bio_module_context_t\s+bio_ctx\s*;',  # bio_ctx
        r'\s*bool\s+bio_async_enabled\s*;',  # bio_async_enabled
        r'\s*nimcp_mutex_t\s*\*?\s*mutex\s*;',  # nimcp_mutex_t* mutex
        r'\s*void\s*\*\s*mutex\s*;',  # void* mutex
        r'\s*pthread_mutex_t\s+mutex\s*;',  # pthread_mutex_t mutex
    ]

    for pattern in fields_to_remove:
        body = re.sub(pattern, '', body)

    # Add bridge_base_t base as first member
    # Find first non-whitespace after opening brace
    body = body.lstrip()
    body = '\n    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */\n    ' + body

    return prefix + body + suffix

def add_accessor_macros(header_content, struct_name):
    """Add accessor macros after struct definition if not present."""
    # Check if accessors already exist
    if '_GET_' in header_content:
        return header_content

    # Find position after struct definition
    struct_end = header_content.find(struct_name + ';')
    if struct_end == -1:
        return header_content

    # Find end of line after struct
    line_end = header_content.find('\n', struct_end)
    if line_end == -1:
        line_end = len(header_content)

    # Generate accessor macro prefix from struct name
    # e.g., swarm_brain_fep_bridge_t -> SWARM_BRAIN_FEP
    prefix = struct_name.replace('_bridge_t', '').replace('_t', '').upper()

    accessor_comment = f'''

/* Accessor macros for type-safe system pointers */
#define {prefix}_GET_SYSTEM_A(bridge) ((void*)(bridge)->base.system_a)
#define {prefix}_GET_SYSTEM_B(bridge) ((void*)(bridge)->base.system_b)
'''

    return header_content[:line_end] + accessor_comment + header_content[line_end:]

def ensure_bridge_base_include_in_header(header_content):
    """Ensure header includes bridge_base.h."""
    if check_has_bridge_base_include(header_content):
        return header_content

    return add_bridge_base_include(header_content)

def process_file(src_path, dry_run=False):
    """Process a single source file and its header."""
    print(f"Processing: {src_path.relative_to(PROJECT_ROOT)}")

    # Read source file
    with open(src_path, 'r') as f:
        src_content = f.read()

    modified_src = False
    modified_header = False

    # 1. Ensure source has bridge_base.h include
    if not check_has_bridge_base_include(src_content):
        src_content = add_bridge_base_include(src_content)
        modified_src = True
        print(f"  - Added bridge_base.h include to source")

    # 2. Fix BRIDGE_DEFINE_BIO_ASYNC_FUNCS_TYPE macro signature
    new_src_content = fix_bio_async_macro(src_content)
    if new_src_content != src_content:
        src_content = new_src_content
        modified_src = True
        print(f"  - Fixed bio-async macro signature")

    # 3. Find and process header
    header_path = find_header_for_source(src_path)
    if header_path:
        with open(header_path, 'r') as f:
            header_content = f.read()

        # 3a. Ensure header has bridge_base.h include
        if not check_has_bridge_base_include(header_content):
            header_content = ensure_bridge_base_include_in_header(header_content)
            modified_header = True
            print(f"  - Added bridge_base.h include to header")

        # 3b. Check if struct needs updating
        if not check_has_bridge_base(header_content):
            struct_name = extract_struct_name(header_content, src_path.name)
            if struct_name:
                header_content = update_struct_in_header(header_content, struct_name)
                header_content = add_accessor_macros(header_content, struct_name)
                modified_header = True
                print(f"  - Updated struct {struct_name} with bridge_base_t")

        # Write header if modified
        if modified_header and not dry_run:
            with open(header_path, 'w') as f:
                f.write(header_content)
            print(f"  - Wrote header: {header_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  - Warning: Could not find header for {src_path.name}")

    # Write source if modified
    if modified_src and not dry_run:
        with open(src_path, 'w') as f:
            f.write(src_content)
        print(f"  - Wrote source: {src_path.relative_to(PROJECT_ROOT)}")

    return modified_src or modified_header

def main():
    parser = argparse.ArgumentParser(description='Refactor bridge files to use bridge_base_t')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without writing')
    args = parser.parse_args()

    print("=" * 60)
    print("Bridge Base Refactoring Script")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")

    print()

    # Find all bridge files
    bridge_files = find_bridge_files()
    print(f"Found {len(bridge_files)} bridge files to process")
    print()

    # Process each file
    modified_count = 0
    for filepath in sorted(bridge_files):
        try:
            if process_file(filepath, args.dry_run):
                modified_count += 1
        except Exception as e:
            print(f"  - Error processing {filepath}: {e}")

    print()
    print("=" * 60)
    print(f"Summary: Modified {modified_count} of {len(bridge_files)} files")
    if args.dry_run:
        print("(Dry run - no files were actually modified)")
    print("=" * 60)

if __name__ == '__main__':
    main()
