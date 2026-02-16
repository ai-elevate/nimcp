#!/usr/bin/env python3
"""
SRP Include-Based Split: Organize god files into #include-based parts
=====================================================================

This script implements the #include "part.c" pattern for SRP splitting:
- The original .c file remains the ONLY compiled unit
- Function groups are extracted into part files
- The parent #includes part files, so they share all types/structs/statics
- No CMakeLists.txt changes needed

Strategy: Split the file at the first function definition.
- Header section: includes, types, structs, globals, forward declarations
- Function section: all function implementations -> distributed to part files

This avoids the opaque struct visibility issue that kills separate compilation,
and avoids orphaned comment fragments.

Usage:
    python3 tools/srp_include_split.py [--dry-run] [--module NAME]
"""

import os
import re
import sys
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"

# Minimum file size to split (lines)
MIN_LINES_TO_SPLIT = 400

# Minimum functions per part file
MIN_FUNCS_PER_PART = 2


def find_function_boundaries(lines):
    """
    Find all function definitions in C source code lines.
    Returns list of (name, start_line, end_line, is_static) tuples.
    start_line includes preceding comment block.
    Lines are 0-indexed.
    """
    functions = []

    # Pattern to match function definition starts
    func_pattern = re.compile(
        r'^(static\s+)?(inline\s+)?'
        r'(?:void|int|bool|float|double|char\s*\*?|const\s+\w+|unsigned\s+\w*|'
        r'uint\d+_t|int\d+_t|size_t|ssize_t|off_t|pid_t|time_t|'
        r'nimcp_\w+|brain_\w+|omni_\w+|nlp_\w+|portia_\w+|health_\w+|'
        r'swarm_\w+|rcog_\w+|fep_\w+|lnn_\w+|mirror_\w+|salience_\w+|'
        r'knowledge_\w+|ethics_\w+|wellbeing_\w+|introspection_\w+|'
        r'plasticity_\w+|language_\w+|collective_\w+|hypergraph_\w+|'
        r'global_\w+|working_\w+|bio_\w+|logging_\w+|corrigibility_\w+|'
        r'distributed_\w+|training_\w+|systems_\w+|adaptive_\w+|'
        r'checkpoint_\w+|deadlock_\w+|runtime_\w+|security_\w+|'
        r'struct\s+\w+\s*\*|'
        r'\w+_t)'  # catch-all for typedef'd types
        r'\s*\*?\s*(\w+)\s*\([^;]*$'
    )

    i = 0
    in_block_comment = False
    while i < len(lines):
        stripped = lines[i].strip()

        # Track block comments
        if '/*' in stripped and '*/' not in stripped:
            in_block_comment = True
            i += 1
            continue
        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            i += 1
            continue

        # Skip preprocessor, line comments, blank
        if stripped.startswith('#') or stripped.startswith('//') or stripped == '':
            i += 1
            continue

        # Skip forward declarations and variable declarations
        if stripped.endswith(';'):
            i += 1
            continue

        # Skip struct/enum/union definitions
        if re.match(r'^(typedef\s+)?(struct|enum|union)\s', stripped):
            # Find matching closing brace
            depth = 0
            j = i
            while j < len(lines):
                depth += lines[j].count('{') - lines[j].count('}')
                if depth == 0 and '{' in ''.join(lines[i:j+1]):
                    break
                j += 1
            # Skip past the semicolon
            while j < len(lines) and ';' not in lines[j]:
                j += 1
            i = j + 1
            continue

        m = func_pattern.match(stripped)
        if m:
            is_static = m.group(1) is not None
            func_name = m.group(3)

            func_start = i  # Start of function signature

            # Find the opening brace - but abort if we find ';' first
            # (that means this is a forward declaration, not a definition)
            brace_line = i
            is_forward_decl = False
            while brace_line < len(lines) and '{' not in lines[brace_line]:
                # Check if line ends with ); or just ; after closing paren
                line_stripped = lines[brace_line].strip()
                if line_stripped.endswith(');') or (');' in line_stripped):
                    is_forward_decl = True
                    break
                brace_line += 1

            if is_forward_decl:
                i = brace_line + 1
                continue

            if brace_line >= len(lines):
                i += 1
                continue

            # Find matching closing brace
            depth = 0
            end_line = brace_line
            for j in range(brace_line, len(lines)):
                depth += lines[j].count('{') - lines[j].count('}')
                if depth == 0:
                    end_line = j
                    break

            functions.append((func_name, func_start, end_line, is_static))
            i = end_line + 1
        else:
            i += 1

    return functions


def find_first_function_line(lines):
    """
    Find the line number of the first function definition.
    Returns 0-indexed line number, or len(lines) if none found.
    We look for the first line that matches a function definition pattern,
    then walk backwards to include its preceding comment/separator block.
    """
    func_pattern = re.compile(
        r'^(static\s+)?(inline\s+)?'
        r'(?:void|int|bool|float|double|char\s*\*?|const\s+\w+|unsigned\s+\w*|'
        r'uint\d+_t|int\d+_t|size_t|ssize_t|off_t|pid_t|time_t|'
        r'nimcp_\w+|brain_\w+|omni_\w+|nlp_\w+|portia_\w+|health_\w+|'
        r'swarm_\w+|rcog_\w+|fep_\w+|lnn_\w+|mirror_\w+|salience_\w+|'
        r'knowledge_\w+|ethics_\w+|wellbeing_\w+|introspection_\w+|'
        r'plasticity_\w+|language_\w+|collective_\w+|hypergraph_\w+|'
        r'global_\w+|working_\w+|bio_\w+|logging_\w+|corrigibility_\w+|'
        r'distributed_\w+|training_\w+|systems_\w+|adaptive_\w+|'
        r'checkpoint_\w+|deadlock_\w+|runtime_\w+|security_\w+|'
        r'struct\s+\w+\s*\*|'
        r'\w+_t)'  # catch-all for typedef'd types
        r'\s*\*?\s*\w+\s*\([^;]*$'
    )

    in_block_comment = False
    in_struct = False
    brace_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track block comments
        if '/*' in stripped and '*/' not in stripped:
            in_block_comment = True
            continue
        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            continue

        # Track struct/enum/union definitions (they have braces too)
        if re.match(r'^(typedef\s+)?(struct|enum|union)\s', stripped):
            in_struct = True
            brace_depth = 0

        if in_struct:
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth == 0 and '{' in stripped:
                in_struct = False
            continue

        # Skip preprocessor, comments, blank, forward declarations
        if stripped.startswith('#') or stripped.startswith('//') or stripped == '' or stripped.endswith(';'):
            continue

        if func_pattern.match(stripped):
            return i

    return len(lines)


def get_split_file_groupings(module_name, parent_dir):
    """
    Read existing split files to determine function groupings.
    Returns dict: {group_name: set(func_names)}
    Only considers split files that DON'T start with 'part_' (those are our output).
    """
    groupings = OrderedDict()
    func_pattern = re.compile(
        r'^(?:static\s+)?(?:inline\s+)?'
        r'(?:void|int|bool|float|double|char|const\s+\w+|unsigned|'
        r'uint\d+_t|int\d+_t|size_t|ssize_t|off_t|pid_t|time_t|'
        r'nimcp_\w+|brain_\w+|omni_\w+|nlp_\w+|portia_\w+|health_\w+|'
        r'swarm_\w+|rcog_\w+|fep_\w+|lnn_\w+|mirror_\w+|salience_\w+|'
        r'knowledge_\w+|ethics_\w+|wellbeing_\w+|introspection_\w+|'
        r'plasticity_\w+|language_\w+|collective_\w+|hypergraph_\w+|'
        r'global_\w+|working_\w+|bio_\w+|logging_\w+|corrigibility_\w+|'
        r'distributed_\w+|training_\w+|systems_\w+|adaptive_\w+|'
        r'checkpoint_\w+|deadlock_\w+|runtime_\w+|security_\w+|'
        r'struct\s+\w+\s*\*|'
        r'\w+_t)'  # catch-all for typedef'd types
        r'\s*\*?\s*(\w+)\s*\('
    )

    for f in sorted(parent_dir.iterdir()):
        if not f.is_file() or f.suffix != '.c':
            continue
        if not f.stem.startswith(module_name + '_'):
            continue
        # Skip our own output files, facades, backups
        if '_part_' in f.stem or 'facade' in f.stem:
            continue
        if f.name.endswith('.orig'):
            continue

        # Derive group name from split file name
        group = f.stem[len(module_name) + 1:]

        text = f.read_text()
        funcs = set()
        for m in func_pattern.finditer(text, re.MULTILINE):
            funcs.add(m.group(1))

        if funcs:
            groupings[group] = funcs

    return groupings


def auto_group_functions(functions, module_name):
    """
    Auto-group functions by common prefix patterns when no split files exist.
    """
    groups = OrderedDict()

    for func_name, start, end, is_static in functions:
        if any(k in func_name for k in ['_init', '_create', '_destroy', '_free', '_cleanup', '_reset', '_shutdown', '_close']):
            group = 'lifecycle'
        elif any(k in func_name for k in ['_update', '_step', '_tick', '_process', '_run', '_execute', '_handle']):
            group = 'processing'
        elif any(k in func_name for k in ['_get_', '_set_', '_is_', '_has_', '_config']):
            group = 'accessors'
        elif any(k in func_name for k in ['_serialize', '_deserialize', '_save', '_load', '_export', '_import', '_read', '_write', '_print', '_dump', '_to_string', '_format']):
            group = 'io'
        elif any(k in func_name for k in ['_stats', '_metric', '_count', '_report', '_log_']):
            group = 'stats'
        elif is_static:
            group = 'helpers'
        else:
            group = 'core'

        if group not in groups:
            groups[group] = []
        groups[group].append(func_name)

    return {k: set(v) for k, v in groups.items()}


def split_module(orig_path, dry_run=False):
    """
    Split a single module using #include-based approach.

    Strategy:
    1. Mark all function definition line ranges
    2. Everything NOT inside a function body = header (kept in parent)
    3. Function bodies = distributed to part files by responsibility
    4. Parent = header (all non-function content) + #include directives

    This ensures types/structs/macros/globals that appear BETWEEN functions
    stay in the parent file where all part files can see them.

    Returns (num_parts, total_funcs_extracted).
    """
    module_name = orig_path.stem.replace('.c', '')
    parent_dir = orig_path.parent

    orig_text = orig_path.read_text()
    orig_lines = orig_text.split('\n')

    if len(orig_lines) < MIN_LINES_TO_SPLIT:
        return 0, 0

    # Find all functions
    functions = find_function_boundaries(orig_lines)
    if len(functions) < 4:
        return 0, 0

    # Build function name -> info map
    func_map = {f[0]: f for f in functions}

    # Get groupings from existing split files
    groupings = get_split_file_groupings(module_name, parent_dir)
    if not groupings:
        groupings = auto_group_functions(functions, module_name)

    # Map functions to groups
    group_funcs = OrderedDict()
    assigned = set()

    for group, func_names in groupings.items():
        members = []
        for fname in func_names:
            if fname in func_map and fname not in assigned:
                members.append(func_map[fname])
                assigned.add(fname)
        if members:
            members.sort(key=lambda x: x[1])
            group_funcs[group] = members

    # Assign unassigned to 'core'
    unassigned = [f for f in functions if f[0] not in assigned]
    if unassigned:
        if 'core' in group_funcs:
            group_funcs['core'].extend(unassigned)
            group_funcs['core'].sort(key=lambda x: x[1])
        else:
            group_funcs['core'] = sorted(unassigned, key=lambda x: x[1])

    # Merge small groups
    merged = OrderedDict()
    small_funcs = []
    for group, members in group_funcs.items():
        if len(members) < MIN_FUNCS_PER_PART:
            small_funcs.extend(members)
        else:
            merged[group] = members

    if small_funcs:
        if merged:
            largest = max(merged, key=lambda k: len(merged[k]))
            merged[largest].extend(small_funcs)
            merged[largest].sort(key=lambda x: x[1])
        else:
            merged['core'] = sorted(small_funcs, key=lambda x: x[1])

    if len(merged) <= 1:
        return 0, 0

    # Build set of lines that belong to function bodies
    func_line_ranges = set()
    for func in functions:
        name, start, end, is_static = func
        for line_no in range(start, end + 1):
            func_line_ranges.add(line_no)

    # For each function, also capture the preceding comment/separator block
    # (lines between previous function end and this function start that are
    # comments or blank lines). These go with the function, not the header.
    all_funcs_sorted = sorted(functions, key=lambda x: x[1])
    func_preamble_ranges = {}  # func_name -> (preamble_start, func_start)

    for idx, func in enumerate(all_funcs_sorted):
        name, start, end, is_static = func
        # Walk backwards from function start to find comment block
        preamble_start = start
        while preamble_start > 0:
            prev = orig_lines[preamble_start - 1].strip()
            if (prev == '' or prev.startswith('//') or prev.startswith('/*') or
                prev.startswith('*') or prev.endswith('*/') or
                prev.startswith('/* ') or prev.startswith('// ')):
                # Check this isn't a type-defining comment block
                # (i.e., the line before the blank/comment is a type definition)
                preamble_start -= 1
            else:
                break

        # Don't capture preamble if it overlaps with a previous function or header content
        if idx > 0:
            prev_func_end = all_funcs_sorted[idx - 1][2]
            if preamble_start <= prev_func_end:
                preamble_start = prev_func_end + 1
            # Check if there's non-comment content between prev_func_end and preamble_start
            # If so, that's header content - don't include it in preamble
            for check_line in range(prev_func_end + 1, preamble_start):
                stripped = orig_lines[check_line].strip()
                if (stripped and not stripped.startswith('//') and
                    not stripped.startswith('/*') and not stripped.startswith('*') and
                    not stripped.endswith('*/')):
                    # There's code between functions - preamble starts after it
                    preamble_start = start
                    break

        func_preamble_ranges[name] = (preamble_start, start)
        # Add preamble lines to function ranges
        for line_no in range(preamble_start, start):
            func_line_ranges.add(line_no)

    # Build header: all lines NOT in any function body or preamble
    header_lines = []
    i = 0
    while i < len(orig_lines):
        if i in func_line_ranges:
            i += 1
        else:
            header_lines.append(orig_lines[i])
            i += 1

    # Clean up excessive blank lines in header
    cleaned_header = []
    blank_count = 0
    for line in header_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned_header.append(line)
        else:
            blank_count = 0
            cleaned_header.append(line)

    # Generate forward declarations for ALL static functions
    # Always generate even if name appears in header (it might be in a macro, not a decl)
    fwd_decls = []
    existing_decls = set()
    # Find existing forward declarations (lines ending with );)
    for line in cleaned_header:
        stripped = line.strip()
        if stripped.startswith('static') and stripped.endswith(');'):
            m = re.search(r'\b(\w+)\s*\(', stripped)
            if m:
                existing_decls.add(m.group(1))
    fwd_decls = []
    for func in functions:
        name, start, end, is_static = func
        if is_static:
            if name in existing_decls:
                continue
            # Build signature by tracking parenthesis depth to handle
            # function pointer parameters like void (*cb)(int, void*)
            sig_line = orig_lines[start].strip()
            paren_depth = sig_line.count('(') - sig_line.count(')')
            j = start + 1
            while j <= end and paren_depth > 0:
                next_part = orig_lines[j].strip()
                sig_line += ' ' + next_part
                paren_depth += next_part.count('(') - next_part.count(')')
                j += 1
            sig_line = re.sub(r'\s*\{.*', '', sig_line).strip()
            if not sig_line.endswith(';'):
                sig_line += ';'
            fwd_decls.append(sig_line)

    if fwd_decls:
        cleaned_header.append('')
        cleaned_header.append('// Forward declarations for static functions (SRP split)')
        for decl in fwd_decls:
            cleaned_header.append(decl)

    # Build function blocks: preamble + function body for each function
    func_blocks = {}
    for func in all_funcs_sorted:
        name, start, end, is_static = func
        preamble_start, func_start = func_preamble_ranges[name]
        block_text = '\n'.join(orig_lines[preamble_start:end + 1])
        func_blocks[name] = block_text

    # Create part files
    current_c = parent_dir / (module_name + '.c')
    total_extracted = 0
    part_files = []

    include_lines = []
    include_lines.append('')
    include_lines.append('//=============================================================================')
    include_lines.append('// SRP Split: Function implementations organized by responsibility')
    include_lines.append('//=============================================================================')

    for group, members in merged.items():
        part_name = f"{module_name}_part_{group}.c"
        part_path = parent_dir / part_name

        part_content = []
        part_content.append(f'// {part_name} - {group} functions')
        part_content.append(f'// Part of {module_name}.c (SRP #include-based split)')
        part_content.append(f'// DO NOT compile separately - #included from {module_name}.c')
        part_content.append('')

        for func in members:
            fname = func[0]
            if fname in func_blocks:
                part_content.append(func_blocks[fname])
                part_content.append('')
                total_extracted += 1

        if not dry_run:
            with open(part_path, 'w') as f:
                f.write('\n'.join(part_content))

        part_files.append((group, part_name, len(members)))
        include_lines.append(f'#include "{part_name}"  // {len(members)} functions: {group}')

    include_lines.append('')

    # Write restructured parent: header + #include directives
    if not dry_run:
        final = '\n'.join(cleaned_header + include_lines)
        with open(current_c, 'w') as f:
            f.write(final)

    return len(part_files), total_extracted


def main():
    dry_run = '--dry-run' in sys.argv
    target_module = None
    for arg in sys.argv[1:]:
        if arg.startswith('--module='):
            target_module = arg.split('=', 1)[1]

    print(f"SRP Include-Based Split {'(DRY RUN)' if dry_run else ''}")
    print("=" * 60)

    orig_files = sorted(SRC_ROOT.rglob("*.c.orig"))
    print(f"Found {len(orig_files)} god file backups\n")

    total_parts = 0
    total_funcs = 0
    modules_split = 0

    for orig in orig_files:
        module_name = orig.stem.replace('.c', '')

        if target_module and target_module not in module_name:
            continue

        current_c = orig.parent / (module_name + '.c')
        if not current_c.exists():
            print(f"  SKIP {module_name}: no current .c file")
            continue

        # Check if original is intact
        orig_size = orig.stat().st_size
        current_size = current_c.stat().st_size
        if abs(orig_size - current_size) > 100:
            print(f"  SKIP {module_name}: current file differs from backup (already modified?)")
            continue

        orig_lines = sum(1 for _ in open(orig))

        num_parts, num_funcs = split_module(orig, dry_run=dry_run)

        if num_parts > 0:
            print(f"  {module_name}: {orig_lines} lines -> {num_parts} parts ({num_funcs} functions)")
            total_parts += num_parts
            total_funcs += num_funcs
            modules_split += 1
        else:
            print(f"  {module_name}: {orig_lines} lines (too small or too few groups)")

    print(f"\n{'=' * 60}")
    print(f"Split {modules_split} modules into {total_parts} part files ({total_funcs} functions)")
    if not dry_run:
        print(f"\nNo CMakeLists.txt changes needed - part files are #included.")
        print(f"Next: cd build && cmake .. && make nimcp -j4")
    else:
        print(f"\nDRY RUN - no files modified")


if __name__ == '__main__':
    main()
