#!/usr/bin/env python3
"""
Fix CRITICAL bug: `return -1` in pointer-returning functions.

On 64-bit systems, `return -1` from a function returning a pointer creates
0xFFFFFFFFFFFFFFFF - a non-NULL garbage pointer that crashes on dereference.

This script scans all .c files under src/cognitive/ and changes `return -1;`
to `return NULL;` ONLY inside functions that return pointer types.

Usage:
    python3 scripts/fix_return_minus1_ptr.py --dry-run   # Preview changes
    python3 scripts/fix_return_minus1_ptr.py              # Apply changes
"""

import os
import re
import sys
import argparse

# Non-pointer return types that legitimately use `return -1`
NON_POINTER_TYPES = {
    'int', 'bool', 'void', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'size_t', 'ssize_t', 'ptrdiff_t',
    'float', 'double',
    'nimcp_error_t', 'error_t',
    'unsigned',  # unsigned int
}

# Qualifiers/storage-class specifiers that precede the actual return type
QUALIFIERS = {'static', 'const', 'volatile', 'inline', 'NIMCP_EXPORT',
              'extern', '__attribute__', '_Thread_local'}


def parse_return_type(line):
    """
    Given a line that looks like a function definition, extract the return type.
    Returns (return_type_str, func_name, is_pointer) or None if not a function def.

    Patterns we handle:
      type* func_name(...)
      type *func_name(...)
      static type* func_name(...)
      static const type* func_name(...)
      NIMCP_EXPORT type* func_name(...)
      struct foo* func_name(...)
      const struct foo* func_name(...)
    """
    # Strip leading whitespace
    stripped = line.lstrip()

    # Skip preprocessor directives
    if stripped.startswith('#'):
        return None

    # Must contain '(' to be a function definition
    if '(' not in stripped:
        return None

    # Get everything before the first '('
    before_paren = stripped.split('(')[0].strip()

    # Must have at least two tokens (return_type + func_name)
    # But need to handle: type* func, type *func, type * func
    # Split on whitespace first
    tokens = before_paren.split()
    if len(tokens) < 2:
        return None

    # The last token (possibly with leading *) is the function name
    func_name = tokens[-1].lstrip('*')

    # Validate function name - must be a valid C identifier
    if not func_name or not re.match(r'^[a-zA-Z_]\w*$', func_name):
        return None

    # Everything before the function name is the return type
    # We need to reconstruct it carefully
    return_part = before_paren[:before_paren.rfind(func_name)].strip()

    # If return_part ends with *, that's part of the return type
    # If the token before func_name had leading *, that's also pointer
    last_token_raw = tokens[-1]
    has_star_prefix = last_token_raw.startswith('*')

    # Check if return type contains a pointer
    is_pointer = '*' in return_part or has_star_prefix

    # Reconstruct full return type string
    full_return_type = return_part
    if has_star_prefix:
        # Stars were attached to the function name token
        stars = ''
        for ch in last_token_raw:
            if ch == '*':
                stars += '*'
            else:
                break
        full_return_type = return_part + stars if return_part else stars

    # Strip qualifiers to get the base type for validation
    base_type = full_return_type
    for q in QUALIFIERS:
        base_type = re.sub(r'\b' + re.escape(q) + r'\b', '', base_type)
    base_type = base_type.replace('*', '').strip()
    # Handle "struct foo" -> "struct_foo" etc.
    base_type = base_type.split()[-1] if base_type.split() else base_type

    return (full_return_type, func_name, is_pointer)


def process_file(filepath, dry_run=True):
    """
    Process a single .c file. Returns list of (line_number, func_name, return_type) changes.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    changes = []
    new_lines = list(lines)  # Copy for modification

    # State machine
    current_func_name = None
    current_return_type = None
    current_is_pointer = False
    brace_depth = 0
    in_function_body = False
    in_block_comment = False
    in_macro_continuation = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- Track block comments ---
        # Handle block comments: could start/end multiple times on one line
        temp_line = stripped
        while True:
            if in_block_comment:
                end_idx = temp_line.find('*/')
                if end_idx >= 0:
                    in_block_comment = False
                    temp_line = temp_line[end_idx + 2:]
                else:
                    break  # Still in block comment
            else:
                start_idx = temp_line.find('/*')
                if start_idx >= 0:
                    # Check if it's not inside a string (rough heuristic: not after odd quotes)
                    in_block_comment = True
                    temp_line = temp_line[start_idx + 2:]
                else:
                    break

        # If we're inside a block comment, skip this line for analysis
        if in_block_comment:
            continue

        # --- Track macro continuations ---
        if in_macro_continuation:
            # Still inside a multi-line #define
            if not stripped.endswith('\\'):
                in_macro_continuation = False
            continue

        if stripped.startswith('#'):
            if stripped.endswith('\\'):
                in_macro_continuation = True
            continue

        # --- Skip single-line comments for return -1 detection ---
        # (but we still need to parse function defs that might have trailing //)
        code_part = stripped
        # Remove string literals first (rough: skip anything between quotes)
        # Then remove // comments
        in_string = False
        escape_next = False
        comment_start = -1
        for ci, ch in enumerate(code_part):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not in_string:
                in_string = True
            elif ch == '"' and in_string:
                in_string = False
            elif not in_string and ch == '/' and ci + 1 < len(code_part) and code_part[ci + 1] == '/':
                comment_start = ci
                break

        if comment_start >= 0:
            code_part = code_part[:comment_start].strip()

        # --- Track brace depth to know when we exit a function ---
        for ch in code_part:
            if ch == '{':
                brace_depth += 1
                if not in_function_body and brace_depth == 1:
                    # Entering function body (or struct, but we only care about
                    # functions we've identified)
                    in_function_body = True
            elif ch == '}':
                brace_depth -= 1
                if brace_depth <= 0:
                    brace_depth = 0
                    in_function_body = False
                    current_func_name = None
                    current_return_type = None
                    current_is_pointer = False

        # --- Detect function definitions ---
        # A function definition starts at brace_depth 0 (before the opening brace)
        # and has a pattern: [qualifiers] type [*] name ( ... ) {
        # We look for this pattern when brace_depth is 0 or 1 (the '{' on same line
        # would have already incremented it)

        # Only try to match function defs when we're at file scope
        # (brace_depth was 0 at start of line, or became 1 due to opening brace on this line)
        line_had_opening_brace = '{' in code_part
        depth_before_line = brace_depth - code_part.count('{') + code_part.count('}')

        if depth_before_line == 0:
            result = parse_return_type(code_part)
            if result:
                ret_type, fname, is_ptr = result
                # This could be a function definition - check if there's a '{' on this line
                # or the next few lines (function body starts)
                # For simplicity, we set the current function info and it becomes active
                # once we see the opening brace
                current_func_name = fname
                current_return_type = ret_type
                current_is_pointer = is_ptr

        # --- Check for `return -1;` inside pointer-returning functions ---
        if current_is_pointer and in_function_body and brace_depth >= 1:
            # Check if this line has `return -1;` (with possible whitespace)
            # But NOT inside a macro or comment (already handled above)
            return_match = re.search(r'\breturn\s+-1\s*;', code_part)
            if return_match:
                # Also verify the original line has it (not just the code_part)
                orig_match = re.search(r'\breturn\s+-1\s*;', line)
                if orig_match:
                    changes.append((i + 1, current_func_name, current_return_type))
                    if not dry_run:
                        # Replace `return -1;` with `return NULL;` in the original line
                        new_lines[i] = re.sub(r'\breturn\s+-1\s*;', 'return NULL;', line, count=1)

    if not dry_run and changes:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return changes


def main():
    parser = argparse.ArgumentParser(
        description='Fix return -1 in pointer-returning functions under src/cognitive/')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Preview changes without modifying files')
    parser.add_argument('--path', default='src/cognitive/',
                        help='Path to scan (default: src/cognitive/)')
    args = parser.parse_args()

    base_path = args.path
    if not os.path.isabs(base_path):
        # Make relative to script's expected location
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_path = os.path.join(project_root, base_path)

    if not os.path.isdir(base_path):
        print(f"ERROR: Directory not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    total_changes = 0
    total_files = 0
    file_changes = []

    for root, dirs, files in sorted(os.walk(base_path)):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            changes = process_file(filepath, dry_run=args.dry_run)
            if changes:
                total_files += 1
                total_changes += len(changes)
                file_changes.append((filepath, changes))

    # Print summary
    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n{'=' * 72}")
    print(f"  fix_return_minus1_ptr.py — {mode}")
    print(f"{'=' * 72}")

    for filepath, changes in file_changes:
        rel_path = os.path.relpath(filepath, os.path.dirname(base_path))
        print(f"\n  {rel_path}:")
        for line_num, func_name, ret_type in changes:
            print(f"    line {line_num:5d}: {func_name}() returns '{ret_type}' — "
                  f"return -1 -> return NULL")

    print(f"\n{'=' * 72}")
    print(f"  Total: {total_changes} changes across {total_files} files")
    print(f"{'=' * 72}\n")

    if args.dry_run and total_changes > 0:
        print("  Run without --dry-run to apply these changes.\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
