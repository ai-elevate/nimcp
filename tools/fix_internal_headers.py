#!/usr/bin/env python3
"""
Fix Internal Headers: Move struct definitions from facade files to internal headers
===================================================================================
The SRP split files need access to struct definitions that were originally in the .c files.
This script extracts struct definitions from the .c.orig backups and adds them to the
internal headers.
"""

import re
import os
from pathlib import Path

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"


def find_struct_definitions(source_text):
    """
    Find all struct/typedef struct definitions in source code.
    Returns list of (name, full_text) tuples.
    """
    structs = []

    # Pattern 1: struct name { ... };
    # Pattern 2: typedef struct { ... } name;
    # Pattern 3: typedef struct name { ... } name;
    # Pattern 4: struct name { ... } (definition with members)

    lines = source_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for struct/union definition start
        if (re.match(r'^\s*(typedef\s+)?(struct|union)\s+\w*\s*\{', stripped) or
            (re.match(r'^\s*(typedef\s+)?(struct|union)\s+\w+\s*$', stripped) and
             i + 1 < len(lines) and lines[i + 1].strip().startswith('{'))):

            # Found a struct definition start
            start = i

            # For the case where { is on the next line
            if '{' not in stripped and i + 1 < len(lines):
                i += 1

            # Find matching closing brace
            brace_depth = 0
            end = i
            for j in range(i, len(lines)):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth == 0 and '{' in ''.join(lines[i:j+1]):
                    end = j
                    break

            # Also capture the trailing typedef name and semicolon
            if end < len(lines) - 1 and not lines[end].strip().endswith(';'):
                end += 1

            # Get full text including any preceding comment
            comment_start = start
            while comment_start > 0:
                prev = lines[comment_start - 1].strip()
                if prev.endswith('*/') or prev.startswith('*') or prev.startswith('/**') or prev.startswith('//'):
                    comment_start -= 1
                elif prev == '':
                    if comment_start > 1 and (lines[comment_start - 2].strip().endswith('*/') or lines[comment_start - 2].strip().startswith('//')):
                        comment_start -= 1
                    else:
                        break
                else:
                    break

            full_text = '\n'.join(lines[comment_start:end + 1])

            # Extract the struct name
            name = None
            # Check for typedef struct name { ... } name_t;
            match = re.search(r'}\s*(\w+)\s*;', full_text)
            if match:
                name = match.group(1)
            else:
                # Check for struct name {
                match = re.search(r'struct\s+(\w+)\s*\{', full_text)
                if match:
                    name = match.group(1)

            if name and not name.startswith('__'):
                structs.append((name, full_text))

            i = end + 1
        else:
            i += 1

    return structs


def find_enum_definitions(source_text):
    """Find typedef enum definitions."""
    enums = []
    lines = source_text.split('\n')
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r'^\s*(typedef\s+)?enum\s+\w*\s*\{', stripped) or \
           (re.match(r'^\s*(typedef\s+)?enum\s+\w+\s*$', stripped) and
            i + 1 < len(lines) and lines[i + 1].strip().startswith('{')):

            start = i
            if '{' not in stripped:
                i += 1

            brace_depth = 0
            end = i
            for j in range(i, len(lines)):
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth == 0 and '{' in ''.join(lines[i:j+1]):
                    end = j
                    break
            if end < len(lines) - 1 and not lines[end].strip().endswith(';'):
                end += 1

            full_text = '\n'.join(lines[start:end + 1])
            match = re.search(r'}\s*(\w+)\s*;', full_text)
            if match:
                enums.append((match.group(1), full_text))

            i = end + 1
        else:
            i += 1
    return enums


def find_defines_and_constants(source_text):
    """Find #define constants and static const declarations."""
    defines = []
    for line in source_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#define') and not stripped.startswith('#define NIMCP_') and \
           not stripped.endswith('_H') and not stripped.endswith('_H_'):
            defines.append(stripped)
        elif re.match(r'^\s*static\s+(const\s+)?\w+\s+\w+\s*=', stripped):
            defines.append(stripped)
    return defines


def fix_internal_header(internal_header_path, orig_file_path):
    """
    Fix an internal header by adding struct definitions from the original file.
    """
    if not orig_file_path.exists():
        print(f"  No .orig backup for {internal_header_path.name}")
        return False

    with open(orig_file_path, 'r') as f:
        orig_text = f.read()

    with open(internal_header_path, 'r') as f:
        header_text = f.read()

    # Find structs in original
    structs = find_struct_definitions(orig_text)
    enums = find_enum_definitions(orig_text)

    if not structs and not enums:
        print(f"  No structs/enums found in {orig_file_path.name}")
        return False

    # Filter out structs that are already in the header
    new_structs = []
    for name, text in structs:
        if name not in header_text:
            new_structs.append((name, text))

    new_enums = []
    for name, text in enums:
        if name not in header_text:
            new_enums.append((name, text))

    if not new_structs and not new_enums:
        print(f"  All structs/enums already in {internal_header_path.name}")
        return False

    # Also extract includes from the original that might be missing
    orig_includes = set()
    header_includes = set()
    for line in orig_text.split('\n'):
        if line.strip().startswith('#include'):
            orig_includes.add(line.strip())
    for line in header_text.split('\n'):
        if line.strip().startswith('#include'):
            header_includes.add(line.strip())

    missing_includes = orig_includes - header_includes

    # Insert struct definitions before the #ifdef __cplusplus / closing guard
    insert_point = header_text.find('#ifdef __cplusplus')
    if insert_point == -1:
        insert_point = header_text.rfind('#endif')

    if insert_point == -1:
        print(f"  Can't find insertion point in {internal_header_path.name}")
        return False

    insertion = "\n"

    # Add missing includes
    if missing_includes:
        insertion += "/* Additional includes from original source */\n"
        for inc in sorted(missing_includes):
            # Skip self-include and internal header
            if '_internal.h' in inc:
                continue
            insertion += f"{inc}\n"
        insertion += "\n"

    # Add struct definitions
    if new_structs:
        insertion += "/* ============================================================================\n"
        insertion += " * Internal Struct Definitions (moved from .c for SRP split access)\n"
        insertion += " * ============================================================================ */\n\n"
        for name, text in new_structs:
            insertion += text + "\n\n"

    # Add enum definitions
    if new_enums:
        insertion += "/* ============================================================================\n"
        insertion += " * Internal Enum Definitions (moved from .c for SRP split access)\n"
        insertion += " * ============================================================================ */\n\n"
        for name, text in new_enums:
            insertion += text + "\n\n"

    new_header = header_text[:insert_point] + insertion + header_text[insert_point:]

    with open(internal_header_path, 'w') as f:
        f.write(new_header)

    print(f"  Fixed: {internal_header_path.name} (+{len(new_structs)} structs, +{len(new_enums)} enums, +{len(missing_includes)} includes)")
    return True


def main():
    dry_run = '--dry-run' in os.sys.argv

    # Find all internal headers and their corresponding .orig files
    internal_headers = sorted(SRC_ROOT.rglob("*_internal.h"))
    fixed = 0

    for ih in internal_headers:
        # Derive the original file name
        # e.g., nimcp_brain_immune_internal.h -> nimcp_brain_immune.c.orig
        module_name = ih.stem.replace('_internal', '')
        orig_path = ih.parent / (module_name + '.c.orig')

        if not orig_path.exists():
            # Try without the nimcp_ prefix mapping
            continue

        print(f"\nProcessing: {ih.relative_to(PROJECT_ROOT)}")
        if not dry_run:
            if fix_internal_header(ih, orig_path):
                fixed += 1
        else:
            print(f"  Would fix from {orig_path.name}")
            fixed += 1

    print(f"\n{'='*50}")
    print(f"Fixed {fixed} internal headers")


if __name__ == '__main__':
    main()
