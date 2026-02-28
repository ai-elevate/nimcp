#!/usr/bin/env python3
"""
Parse build output for conflicting type errors and make headers match source.
The source definition is authoritative.

Strategy: For each error, we know:
- func_name: the conflicting function
- header_file:header_line: where the header declaration is
- source_has_const: whether the source has const on the first type_t* param

We read the header, find the exact line(s) with the function declaration,
and fix just that declaration.
"""

import re
import sys

Q = '[\u2018\u2019\'\u201c\u201d"]'  # GCC curly quotes


def parse_build_file(build_file):
    """Parse build output and return list of fixes needed."""
    with open(build_file, 'r') as f:
        lines = f.readlines()

    fixes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "error: conflicting types for" in line:
            func_m = re.search(r'conflicting types for\s+' + Q + r'(\w+)' + Q, line)
            if not func_m:
                i += 1
                continue
            func_name = func_m.group(1)

            # Check if source has const: look at "have '...(const type_t *)...'"
            # The "have" part is the SOURCE definition
            source_sig = re.search(r'have\s+' + Q + r'(.+?)' + Q, line)
            source_has_const = False
            if source_sig:
                source_has_const = 'const ' in source_sig.group(1) and '_t' in source_sig.group(1)

            # Find note line with header info
            for j in range(i+1, min(i+6, len(lines))):
                note_line = lines[j]
                if "previous declaration" in note_line and func_name in note_line:
                    header_m = re.match(r'(\S+\.h):(\d+)', note_line)
                    if header_m:
                        # Extract what the header currently has from "with type '...'"
                        header_sig = re.search(r'with type\s+' + Q + r'(.+?)' + Q, note_line)
                        header_has_const = False
                        if header_sig:
                            header_has_const = 'const ' in header_sig.group(1) and '_t' in header_sig.group(1)

                        fixes.append({
                            'func_name': func_name,
                            'header_file': header_m.group(1),
                            'header_line': int(header_m.group(2)),
                            'source_has_const': source_has_const,
                            'header_has_const': header_has_const,
                        })
                    break
        i += 1
    return fixes


def fix_declaration(file_lines, header_line_num, func_name, source_has_const, header_has_const):
    """Fix a single function declaration in the header to match source."""
    # Collect the full declaration (may span multiple lines)
    start_idx = header_line_num - 1
    if start_idx < 0 or start_idx >= len(file_lines):
        return False

    # Find the line range containing this declaration
    decl_start = start_idx
    decl_end = start_idx

    # Expand to find the closing ');'
    for k in range(start_idx, min(len(file_lines), start_idx + 10)):
        if ');' in file_lines[k]:
            decl_end = k
            break

    # Also look backward if the function name is not on start_idx
    if func_name not in file_lines[start_idx]:
        for k in range(start_idx - 1, max(-1, start_idx - 5), -1):
            if func_name in file_lines[k]:
                decl_start = k
                break

    # Extract the full declaration
    decl_lines = file_lines[decl_start:decl_end + 1]
    full_decl = ''.join(decl_lines)

    if func_name not in full_decl:
        return False

    # Find ALL type_t* params in this declaration
    # We need to find WHICH param has the mismatch
    # Strategy: find the first type_t* param that has/doesn't have const

    if source_has_const and not header_has_const:
        # Need to ADD const to the first non-const type_t* param
        # Pattern: find the first type_t * that doesn't have const before it
        new_decl = re.sub(
            r'(?<!\bconst )(\b\w+_t\s*\*)',
            r'const \1',
            full_decl,
            count=1
        )
    elif not source_has_const and header_has_const:
        # Need to REMOVE const from the first const type_t* param
        new_decl = re.sub(
            r'\bconst\s+(\w+_t\s*\*)',
            r'\1',
            full_decl,
            count=1
        )
    else:
        return False  # Already matches

    if new_decl == full_decl:
        return False

    # Write back the fixed lines
    new_decl_lines = new_decl.split('\n')
    # Preserve line endings
    for k in range(len(new_decl_lines)):
        if k < len(new_decl_lines) - 1:
            new_decl_lines[k] += '\n'

    # Replace the original lines
    file_lines[decl_start:decl_end + 1] = new_decl_lines

    return True


def main():
    build_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/build_output2.txt'
    print(f"Parsing {build_file}...")
    fixes = parse_build_file(build_file)

    # Dedup by (func_name, header_file)
    seen = set()
    unique = []
    for f in fixes:
        key = (f['func_name'], f['header_file'])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    fixes = unique

    print(f"Found {len(fixes)} fixes needed")

    # Show summary
    add_const = sum(1 for f in fixes if f['source_has_const'] and not f['header_has_const'])
    rm_const = sum(1 for f in fixes if not f['source_has_const'] and f['header_has_const'])
    no_change = sum(1 for f in fixes if f['source_has_const'] == f['header_has_const'])
    print(f"  Add const to header: {add_const}")
    print(f"  Remove const from header: {rm_const}")
    print(f"  Already matching (other mismatch): {no_change}")

    # Group by file
    by_file = {}
    for f in fixes:
        by_file.setdefault(f['header_file'], []).append(f)

    total_fixed = 0
    files_modified = set()

    for header_file, entries in sorted(by_file.items()):
        try:
            with open(header_file, 'r') as f:
                file_lines = f.readlines()
        except Exception as e:
            print(f"  ERROR reading {header_file}: {e}")
            continue

        modified = False
        # Process in reverse order of line number to avoid offset issues
        entries.sort(key=lambda x: x['header_line'], reverse=True)

        for entry in entries:
            if entry['source_has_const'] == entry['header_has_const']:
                print(f"  SKIP (both {'have' if entry['source_has_const'] else 'lack'} const): "
                      f"{entry['func_name']} in {header_file}:{entry['header_line']}")
                continue

            if fix_declaration(file_lines, entry['header_line'], entry['func_name'],
                             entry['source_has_const'], entry['header_has_const']):
                action = "added const" if entry['source_has_const'] else "removed const"
                print(f"  FIXED ({action}): {entry['func_name']} in {header_file}:{entry['header_line']}")
                modified = True
                total_fixed += 1
            else:
                print(f"  SKIP (couldn't fix): {entry['func_name']} in {header_file}:{entry['header_line']}")

        if modified:
            with open(header_file, 'w') as f:
                f.writelines(file_lines)
            files_modified.add(header_file)

    print(f"\nFixed {total_fixed} declarations in {len(files_modified)} files")


if __name__ == '__main__':
    main()
