#!/usr/bin/env python3
"""
fix_alloc_overflow.py - Fix integer overflow in allocation size calculations.

Scans .c files in src/cognitive/ for nimcp_malloc/malloc calls where the size
argument contains multiplication (a * b * sizeof(...)), and replaces them with
nimcp_calloc/calloc to let calloc handle the overflow check internally.

Strategy:
  - nimcp_malloc(A * B * sizeof(T))  -> nimcp_calloc((size_t)A * B, sizeof(T))
  - nimcp_malloc(A * sizeof(T))  where A is variable -> nimcp_calloc(A, sizeof(T))
  - malloc(...) same patterns -> calloc(...)
  - nimcp_calloc(count, size) already safe -> SKIP
  - Literal-only multiplications like nimcp_malloc(10 * sizeof(char*)) -> SKIP
  - memset(..., 0, ...) immediately after converted malloc -> remove

Usage:
  python3 scripts/fix_alloc_overflow.py --dry-run   # Preview changes
  python3 scripts/fix_alloc_overflow.py              # Apply changes
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional


class Replacement(NamedTuple):
    file: str
    line_num: int
    old_text: str
    new_text: str
    reason: str


# Match sizeof(type) or sizeof(expr) — handles nested parens inside sizeof
SIZEOF_RE = re.compile(r'sizeof\s*\([^)]+\)')

# Match integer literal
INT_LITERAL_RE = re.compile(r'^\d+$')


def find_malloc_call(line: str):
    """
    Find a nimcp_malloc(...) or malloc(...) call in a line, handling nested parens.
    Returns (start_of_funcname, func_name, args_str, end_pos) or None.
    """
    for func_name in ('nimcp_malloc', 'malloc'):
        # Find the function name (not preceded by alphanumeric to avoid partial matches)
        idx = 0
        while True:
            pos = line.find(func_name, idx)
            if pos == -1:
                break

            # Check it's not part of a longer identifier
            if pos > 0 and (line[pos - 1].isalnum() or line[pos - 1] == '_'):
                # e.g., 'nimcp_calloc' contains 'alloc' but we wouldn't match 'malloc' here
                # For 'nimcp_malloc', check if preceded by something else
                if func_name == 'malloc' and pos >= 6 and line[pos-6:pos] == 'nimcp_':
                    idx = pos + len(func_name)
                    continue
                elif func_name == 'malloc':
                    idx = pos + len(func_name)
                    continue

            # Find the opening paren
            after_name = pos + len(func_name)
            paren_pos = after_name
            while paren_pos < len(line) and line[paren_pos] in ' \t':
                paren_pos += 1

            if paren_pos >= len(line) or line[paren_pos] != '(':
                idx = pos + len(func_name)
                continue

            # Now find the matching closing paren
            depth = 1
            i = paren_pos + 1
            while i < len(line) and depth > 0:
                if line[i] == '(':
                    depth += 1
                elif line[i] == ')':
                    depth -= 1
                i += 1

            if depth != 0:
                # Unbalanced parens (multiline call) - skip
                idx = pos + len(func_name)
                continue

            args_str = line[paren_pos + 1:i - 1]
            return pos, func_name, args_str, i

            idx = pos + len(func_name)

    return None


def split_top_level_multiply(expr: str) -> list:
    """
    Split expression on top-level '*' operators (not inside parens).
    Returns list of factor strings. Distinguishes multiply from pointer deref.
    """
    factors = []
    depth = 0
    current = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '*' and depth == 0:
            # Determine if this is multiplication or pointer dereference
            before = ''.join(current).strip()
            if before and (before[-1].isalnum() or before[-1] in ')]}'):
                # Multiplication operator
                factors.append(before)
                current = []
                i += 1
                continue
            else:
                # Pointer dereference (e.g., *num_patterns)
                current.append(ch)
        else:
            current.append(ch)
        i += 1
    remainder = ''.join(current).strip()
    if remainder:
        factors.append(remainder)
    return factors


def is_literal_or_small_constant(expr: str) -> bool:
    """Check if expression is a literal integer."""
    expr = expr.strip()
    return bool(INT_LITERAL_RE.match(expr))


def classify_and_fix(func_name: str, args_str: str):
    """
    Classify a malloc call and return (new_call, reason) or None if no fix needed.
    """
    args_str = args_str.strip()

    # Determine the calloc function name
    calloc_func = 'nimcp_calloc' if func_name == 'nimcp_malloc' else 'calloc'

    # Split into top-level multiplication factors
    factors = split_top_level_multiply(args_str)

    if len(factors) < 2:
        # No multiplication at all - nothing to fix
        return None

    # Separate sizeof factors from non-sizeof factors
    sizeof_factors = []
    non_sizeof_factors = []
    for f in factors:
        f_stripped = f.strip()
        if re.match(r'^sizeof\s*\(', f_stripped):
            sizeof_factors.append(f_stripped)
        else:
            non_sizeof_factors.append(f_stripped)

    if not sizeof_factors:
        # No sizeof in the multiplication - unusual pattern, skip
        return None

    if not non_sizeof_factors:
        # Only sizeof factors - skip
        return None

    # Count how many non-sizeof factors are variable (non-literal)
    variable_factors = [f for f in non_sizeof_factors if not is_literal_or_small_constant(f)]

    if len(variable_factors) == 0:
        # All non-sizeof factors are literals (e.g., nimcp_malloc(10 * sizeof(char*)))
        # No overflow risk from literals alone
        return None

    # Build the sizeof part (might be multiple, though unlikely)
    sizeof_part = ' * '.join(sizeof_factors)

    if len(non_sizeof_factors) == 1:
        # Pattern: var * sizeof(T) -> calloc(var, sizeof(T))
        count_expr = non_sizeof_factors[0]
        new_call = f'{calloc_func}({count_expr}, {sizeof_part})'
        return new_call, f'single-var-mult: {func_name}({count_expr} * {sizeof_part}) -> {calloc_func}(count, size)'

    if len(non_sizeof_factors) >= 2:
        # Pattern: A * B * sizeof(T) -> calloc((size_t)A * B, sizeof(T))
        # HIGH RISK - this is the primary target
        count_expr = ' * '.join(non_sizeof_factors)
        # Add size_t cast to ensure the multiplication is done in size_t space
        if '(size_t)' not in count_expr:
            count_expr = f'(size_t){count_expr}'
        new_call = f'{calloc_func}({count_expr}, {sizeof_part})'
        reason = f'multi-var-mult: {func_name}({" * ".join(non_sizeof_factors)} * {sizeof_part}) -> {calloc_func}(count, size)'
        return new_call, reason

    return None


def find_memset_zero_after(lines: list, malloc_line_idx: int, var_name: str):
    """
    Check if there's a memset(var, 0, ...) within a window after the malloc.
    Handles the common pattern:
       var = malloc(...)
       if (!var) { ... error handling ... }
       memset(var, 0, ...)

    Returns the line index of the memset, or None.
    """
    escaped = re.escape(var_name)
    # Use .+ instead of [^)]+ to handle nested parens like sizeof(float)
    # The overall match is anchored to line start+end so it's still safe
    memset_pattern = re.compile(
        r'^\s*memset\s*\(\s*' + escaped + r'\s*,\s*0(?:x0+)?\s*,\s*.+\)\s*;\s*$'
    )

    # Scan up to 20 lines ahead (error handling blocks can be long)
    brace_depth = 0
    in_error_block = False

    for offset in range(1, 20):
        idx = malloc_line_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx].strip()

        # Track brace depth for if-blocks
        brace_depth += line.count('{') - line.count('}')

        # Skip blank lines and comments
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
            continue

        # Inside an error-handling block (e.g., if (!var) { ... })
        if brace_depth > 0:
            continue

        # Match memset zeroing our variable
        if memset_pattern.match(line):
            return idx

        # Match: if (!var) or if (var == NULL) followed by single-line return
        if line.startswith('if') and ('!' + var_name in line or var_name in line):
            if '{' not in line:
                # Single-line if; skip this and the next line
                continue
            else:
                in_error_block = True
                continue

        # Another malloc for a different variable - stop looking
        if 'malloc' in line and var_name not in line:
            break

        # A non-memset statement at depth 0 that uses our var (e.g., xavier_init)
        # is fine to skip past
        if brace_depth == 0 and var_name in line:
            continue

        # Any other top-level statement - stop (we've gone too far)
        if brace_depth == 0 and line and line != '}':
            break

    return None


def extract_assigned_var(line: str, func_start: int) -> Optional[str]:
    """Extract the variable name being assigned from the part before the malloc call."""
    prefix = line[:func_start].rstrip()
    # Match: var = (cast), var = , foo->bar = (cast), etc.
    # Strip trailing cast
    prefix = re.sub(r'\([^)]*\)\s*$', '', prefix).rstrip()
    # Now match the assignment
    m = re.search(r'([\w][\w\-\>\.\[\]]*)\s*=\s*$', prefix)
    if m:
        return m.group(1)
    return None


def process_file(filepath: str, dry_run: bool) -> list:
    """Process a single C file and return list of replacements made."""
    replacements = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (OSError, IOError) as e:
        print(f"  WARNING: Cannot read {filepath}: {e}", file=sys.stderr)
        return []

    memset_lines_to_remove = set()
    line_replacements = {}

    for i, line in enumerate(lines):
        result = find_malloc_call(line)
        if result is None:
            continue

        func_start, func_name, args_str, call_end = result

        fix = classify_and_fix(func_name, args_str)
        if fix is None:
            continue

        new_call, reason = fix

        # Build the old call string as it appears in the line
        old_call_str = line[func_start:call_end]
        new_line = line[:func_start] + new_call + line[call_end:]

        # Check if there's a redundant memset(var, 0, ...) after this line
        var_name = extract_assigned_var(line, func_start)
        if var_name:
            memset_idx = find_memset_zero_after(lines, i, var_name)
            if memset_idx is not None:
                memset_lines_to_remove.add(memset_idx)

        line_replacements[i] = new_line
        replacements.append(Replacement(
            file=filepath,
            line_num=i + 1,
            old_text=line.rstrip(),
            new_text=new_line.rstrip(),
            reason=reason
        ))

    # Add memset removals to the replacements list
    for idx in sorted(memset_lines_to_remove):
        replacements.append(Replacement(
            file=filepath,
            line_num=idx + 1,
            old_text=lines[idx].rstrip(),
            new_text='    /* calloc zero-initializes */',
            reason='redundant memset after calloc conversion'
        ))

    if not dry_run and (line_replacements or memset_lines_to_remove):
        new_lines = []
        for i, line in enumerate(lines):
            if i in line_replacements:
                new_lines.append(line_replacements[i])
            elif i in memset_lines_to_remove:
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + '/* calloc zero-initializes */\n')
            else:
                new_lines.append(line)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return replacements


def main():
    parser = argparse.ArgumentParser(
        description='Fix integer overflow in allocation size calculations in src/cognitive/'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output for each replacement'
    )
    parser.add_argument(
        '--path', default=None,
        help='Override scan path (default: src/cognitive/ relative to script location)'
    )
    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.path:
        scan_dir = Path(args.path)
    else:
        scan_dir = project_root / 'src' / 'cognitive'

    if not scan_dir.is_dir():
        print(f"ERROR: Directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Scanning {scan_dir} for unsafe allocation patterns...")
    print()

    c_files = sorted(scan_dir.rglob('*.c'))

    all_replacements = []
    files_changed = 0

    multi_var_count = 0
    single_var_count = 0
    memset_removed_count = 0

    for filepath in c_files:
        replacements = process_file(str(filepath), args.dry_run)
        if replacements:
            files_changed += 1
            all_replacements.extend(replacements)

            for r in replacements:
                if 'multi-var-mult' in r.reason:
                    multi_var_count += 1
                elif 'single-var-mult' in r.reason:
                    single_var_count += 1
                elif 'memset' in r.reason:
                    memset_removed_count += 1

    # Print results
    if args.verbose or args.dry_run:
        by_file = {}
        for r in all_replacements:
            by_file.setdefault(r.file, []).append(r)

        for filepath, reps in sorted(by_file.items()):
            rel_path = os.path.relpath(filepath, project_root)
            print(f"--- {rel_path} ---")
            for r in reps:
                print(f"  L{r.line_num}: [{r.reason}]")
                print(f"    - {r.old_text.strip()}")
                print(f"    + {r.new_text.strip()}")
                print()

    # Summary
    alloc_fixes = multi_var_count + single_var_count
    print("=" * 70)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Files scanned:       {len(c_files)}")
    print(f"  Files modified:      {files_changed}")
    print(f"  Allocation fixes:    {alloc_fixes}")
    print(f"    HIGH risk (A*B*sizeof): {multi_var_count}")
    print(f"    MEDIUM risk (var*sizeof): {single_var_count}")
    print(f"  Redundant memset removed: {memset_removed_count}")
    print(f"  Total changes:       {len(all_replacements)}")

    if args.dry_run and all_replacements:
        print()
        print("Run without --dry-run to apply these changes.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
