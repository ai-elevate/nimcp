#!/usr/bin/env python3
"""
fix_switch_fallthrough.py - Find and fix unintentional switch fallthrough in C files.

Scans all .c files under src/cognitive/ for switch case labels where the preceding
case body has executable statements but does NOT end with break/return/goto/continue
or a fallthrough comment.

CONSERVATIVE approach:
  - Only adds `break;` where it is CLEARLY a bug
  - Skips stacked cases (case A: case B: with no code between)
  - Skips cases terminated by return/goto/continue/break (even multi-line)
  - Skips fallthrough comments
  - Skips default: labels (often intentional)
  - Skips cases where previous case body ends with `}` on its own line
    (often closing an if/for/while block with all paths covered)
  - Handles multi-line return/function-call statements correctly
  - Handles `func(); break;` on a single line

Usage:
    python3 scripts/fix_switch_fallthrough.py --dry-run    # Count only
    python3 scripts/fix_switch_fallthrough.py              # Apply fixes
    python3 scripts/fix_switch_fallthrough.py --verbose    # Show each fix
"""

import argparse
import os
import re
import sys
from pathlib import Path


# Pattern for case/default labels
CASE_LABEL_RE = re.compile(r'^\s*case\s+[A-Za-z_0-9]+\s*:')
DEFAULT_LABEL_RE = re.compile(r'^\s*default\s*:')

# Pattern for fallthrough comments
FALLTHROUGH_COMMENT_RE = re.compile(
    r'/\*\s*(?:fall\s*through|FALL\s*THROUGH|FALLTHROUGH|falls?\s*thro?u?gh)\s*\*/'
    r'|//\s*(?:fall\s*through|FALL\s*THROUGH|FALLTHROUGH|falls?\s*thro?u?gh)',
    re.IGNORECASE
)

# Pattern for opening brace of switch
SWITCH_OPEN_RE = re.compile(r'^\s*switch\s*\(')

# Flow control keywords that terminate a case body
FLOW_CONTROL_KEYWORDS = {'break', 'return', 'goto', 'continue', 'exit', '_exit', 'abort'}

# Patterns for flow control at the END of a (possibly multi-line) statement:
# We check if the line itself (or a line ending the same logical statement)
# contains break;, return ...;, goto ...;, continue;, exit();, abort();
FLOW_CONTROL_END_RE = re.compile(
    r'(?:^|\s|;)'  # preceded by start, whitespace, or semicolon
    r'(?:'
    r'break\s*;'
    r'|return\b[^;]*;'
    r'|goto\s+\w+\s*;'
    r'|continue\s*;'
    r'|exit\s*\([^)]*\)\s*;'
    r'|_exit\s*\([^)]*\)\s*;'
    r'|abort\s*\(\s*\)\s*;'
    r')'
    r'\s*(?://.*)?$'  # optional trailing comment
)


def is_case_or_default(line):
    """Return True if line is a case or default label."""
    return bool(CASE_LABEL_RE.match(line)) or bool(DEFAULT_LABEL_RE.match(line))


def is_case_label(line):
    """Return True if line is specifically a 'case X:' label (not default)."""
    return bool(CASE_LABEL_RE.match(line))


def has_fallthrough_comment(line):
    """Return True if line contains a fallthrough annotation comment."""
    return bool(FALLTHROUGH_COMMENT_RE.search(line))


def get_indentation(line):
    """Extract leading whitespace from a line."""
    match = re.match(r'^(\s*)', line)
    return match.group(1) if match else ''


def strip_comments(line):
    """Strip C-style single-line comments from a line (approximate)."""
    # Remove // comments (but not inside strings - approximate)
    result = re.sub(r'//.*$', '', line)
    # Remove /* ... */ single-line comments
    result = re.sub(r'/\*.*?\*/', '', result)
    return result


def is_blank_or_comment(line):
    """Return True if line is blank, whitespace-only, or a comment."""
    stripped = line.strip()
    if stripped == '':
        return True
    if stripped.startswith('//'):
        return True
    if stripped.startswith('/*') and stripped.endswith('*/'):
        return True
    if stripped.startswith('/*'):
        return True
    if stripped.startswith('*'):
        return True
    if stripped.startswith('*/'):
        return True
    return False


def find_statement_block(lines, end_idx):
    """
    Starting from end_idx, walk backwards to find the full logical statement
    that ends at this line. Returns the range (start_idx, end_idx) inclusive
    of the complete statement, and the concatenated text of all significant
    lines in that statement.

    This handles multi-line statements like:
        return foo(
            bar, baz,
            qux);

    Strategy: Walk backwards from end_idx. If the line doesn't start a
    statement (i.e., doesn't begin at zero indent relative to the case or
    with a keyword), it's a continuation. Keep going until we find the
    statement start.
    """
    if end_idx < 0 or end_idx >= len(lines):
        return end_idx, end_idx, ''

    # Collect lines of the statement (bottom-up)
    stmt_lines = []
    idx = end_idx

    # The end line's stripped text
    end_stripped = strip_comments(lines[end_idx]).strip()
    stmt_lines.append(end_stripped)

    # Walk backwards looking for the start of the statement
    # A statement continuation is indicated by:
    # - Line ends with an operator (+, -, |, &, ||, &&, comma, open paren)
    # - OR the next line starts with an operator or continuation token
    # - OR we're inside unclosed parentheses
    paren_depth = end_stripped.count(')') - end_stripped.count('(')
    # If paren_depth > 0, we have more closing parens than opening -> multi-line

    idx -= 1
    while idx >= 0:
        stripped = strip_comments(lines[idx]).strip()

        # Stop at blank/comment lines (statement boundary)
        if is_blank_or_comment(lines[idx]):
            break

        # Stop at case/default labels
        if is_case_or_default(lines[idx]):
            break

        # Stop at preprocessor directives
        if stripped.startswith('#'):
            break

        # Stop at opening/closing braces standing alone
        if stripped == '{' or stripped == '}':
            break

        # Track parentheses (backwards: ')' adds depth, '(' subtracts)
        paren_depth += stripped.count(')') - stripped.count('(')

        # If this line is a continuation of the statement below
        # Heuristic: if paren_depth > 0 or previous significant line ends with
        # a continuation character
        is_continuation = False

        if paren_depth > 0:
            is_continuation = True
        elif stripped.endswith(',') or stripped.endswith('|') or \
             stripped.endswith('&') or stripped.endswith('+') or \
             stripped.endswith('-') or stripped.endswith('\\') or \
             stripped.endswith('(') or stripped.endswith('?') or \
             stripped.endswith(':') and not is_case_or_default(lines[idx]):
            is_continuation = True

        if is_continuation:
            stmt_lines.append(stripped)
            idx -= 1
        else:
            # Check if this line itself might be the beginning of the statement
            # by checking whether it starts with a flow control keyword
            # that could span to the end_idx line
            for kw in FLOW_CONTROL_KEYWORDS:
                if stripped.startswith(kw + ' ') or stripped.startswith(kw + '(') or \
                   stripped.startswith(kw + ';') or stripped == kw:
                    # This is the start of a multi-line flow control statement
                    stmt_lines.append(stripped)
                    idx -= 1
                    paren_depth = 0  # Reset - we found the keyword
                    break
            else:
                break

    # stmt_lines is in reverse order, join them
    stmt_lines.reverse()
    full_stmt = ' '.join(stmt_lines)
    start_idx = idx + 1

    return start_idx, end_idx, full_stmt


def statement_has_flow_control(stmt_text, last_line_text):
    """
    Check if a statement (possibly multi-line, concatenated) contains
    flow control that terminates the case body.

    Also checks the raw last_line_text for patterns like `func(); break;`
    """
    stripped = stmt_text.strip()

    # Check for break/return/goto/continue as the start of the statement
    for kw in ('break', 'return', 'goto', 'continue', 'exit', '_exit', 'abort'):
        if stripped.startswith(kw + ' ') or stripped.startswith(kw + '(') or \
           stripped.startswith(kw + ';') or stripped == kw + ';' or stripped == kw:
            return True

    # Check for break at the end of the line (e.g., `func(); break;`)
    # This covers the pattern: code; break;
    last_stripped = strip_comments(last_line_text).strip()
    if re.search(r'\bbreak\s*;\s*$', last_stripped):
        return True
    if re.search(r'\breturn\b[^;]*;\s*$', last_stripped):
        return True
    if re.search(r'\bgoto\s+\w+\s*;\s*$', last_stripped):
        return True
    if re.search(r'\bcontinue\s*;\s*$', last_stripped):
        return True

    # Check if NIMCP_THROW_TO_IMMUNE appears (often serves as flow control)
    if 'NIMCP_THROW_TO_IMMUNE' in stripped:
        return True

    return False


def find_preceding_significant_line(lines, case_line_idx):
    """
    Walk backwards from case_line_idx to find the last significant
    (non-blank, non-comment) line before this case label.

    Returns (line_index, line_text) or (None, None) if nothing found.
    """
    idx = case_line_idx - 1
    in_block_comment = False

    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()

        # Track block comments (walking backwards)
        if stripped.endswith('*/') and not stripped.startswith('//'):
            if '/*' in stripped:
                # Single-line block comment
                idx -= 1
                continue
            else:
                in_block_comment = True
                idx -= 1
                continue

        if in_block_comment:
            if '/*' in stripped:
                in_block_comment = False
            idx -= 1
            continue

        # Skip blank lines and line comments
        if stripped == '' or stripped.startswith('//'):
            idx -= 1
            continue

        # Skip lines that are just continuation comment markers
        if stripped.startswith('*') and not stripped.startswith('*/'):
            idx -= 1
            continue

        return idx, line

    return None, None


def analyze_file(filepath, dry_run=True, verbose=False):
    """
    Analyze a single C file for unintentional switch fallthrough.

    Returns list of (line_number, context_description) for fixes applied/found.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        if verbose:
            print(f"  WARNING: Could not read {filepath}: {e}", file=sys.stderr)
        return []

    fixes = []
    insertions = []  # (line_index, indentation, description)

    for i, line in enumerate(lines):
        # We only care about `case X:` labels (skip default: as per spec)
        if not is_case_label(line):
            continue

        # Find the preceding significant line
        prev_idx, prev_line = find_preceding_significant_line(lines, i)
        if prev_idx is None:
            continue

        prev_stripped = prev_line.strip()

        # SKIP: If previous significant line is another case/default label
        # (stacked cases -- intentional)
        if is_case_or_default(prev_line):
            continue

        # SKIP: If previous line is just `{` (opening of switch or block)
        if prev_stripped == '{':
            continue

        # SKIP: If previous line is `}` (closing brace -- often intentional;
        # the block inside likely has its own flow control on all paths)
        if prev_stripped.startswith('}'):
            continue

        # SKIP: If previous line is a preprocessor directive
        if prev_stripped.startswith('#'):
            continue

        # Check for fallthrough comments between prev_idx and current case label
        has_ft_between = False
        for j in range(prev_idx, i):
            if has_fallthrough_comment(lines[j]):
                has_ft_between = True
                break
        if has_ft_between:
            continue

        # Now reconstruct the full statement ending at prev_idx
        stmt_start, stmt_end, full_stmt = find_statement_block(lines, prev_idx)

        # Check if the statement has flow control
        if statement_has_flow_control(full_stmt, prev_line):
            continue

        # SKIP: If the statement is not clearly a statement
        # (doesn't end with `;` or `)`)
        code_stripped = strip_comments(prev_line).strip()
        if not (code_stripped.endswith(';') or code_stripped.endswith(')') or
                code_stripped.endswith(',')):
            continue

        # Verify there is actually an owning case/default label above
        found_owning_case = False
        brace_depth = 0
        check_idx = stmt_start
        while check_idx >= 0:
            check_stripped = lines[check_idx].strip()

            # Count braces to track nesting
            brace_depth += check_stripped.count('}') - check_stripped.count('{')

            if brace_depth > 0:
                break

            if is_case_or_default(lines[check_idx]) and brace_depth <= 0:
                found_owning_case = True
                break

            if SWITCH_OPEN_RE.match(lines[check_idx]):
                break

            check_idx -= 1

        if not found_owning_case:
            continue

        # Determine indentation for break; (match the preceding statement)
        indent = get_indentation(prev_line)

        # Record the fix
        line_num = i + 1  # 1-indexed
        prev_line_num = prev_idx + 1
        desc = f"Line {line_num}: missing break; after line {prev_line_num}: {prev_stripped[:80]}"
        fixes.append((line_num, desc))

        insert_at = prev_idx + 1
        insertions.append((insert_at, indent, desc))

    if not dry_run and insertions:
        # Apply insertions in reverse order so line indices remain valid
        insertions.sort(key=lambda x: x[0], reverse=True)
        for insert_idx, indent, desc in insertions:
            break_line = f"{indent}break;\n"
            lines.insert(insert_idx, break_line)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except (IOError, OSError) as e:
            print(f"  ERROR: Could not write {filepath}: {e}", file=sys.stderr)
            return []

    return fixes


def main():
    parser = argparse.ArgumentParser(
        description='Find and fix unintentional switch fallthrough in C files'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report, do not modify files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show each fix location')
    parser.add_argument('--path', default=None,
                        help='Override scan path (default: src/cognitive/)')
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

    mode = "DRY RUN" if args.dry_run else "APPLYING FIXES"
    print(f"=== Switch Fallthrough Fixer ({mode}) ===")
    print(f"Scanning: {scan_dir}")
    print()

    # Collect all .c files
    c_files = sorted(scan_dir.rglob('*.c'))
    print(f"Found {len(c_files)} .c files to scan")
    print()

    total_fixes = 0
    files_with_fixes = 0
    all_fixes = []

    for filepath in c_files:
        fixes = analyze_file(str(filepath), dry_run=args.dry_run, verbose=args.verbose)

        if fixes:
            files_with_fixes += 1
            total_fixes += len(fixes)
            rel_path = filepath.relative_to(project_root)

            if args.verbose:
                print(f"  {rel_path}: {len(fixes)} fallthrough(s)")
                for line_num, desc in fixes:
                    print(f"    {desc}")
                print()
            else:
                print(f"  {rel_path}: {len(fixes)} fallthrough(s)")

            all_fixes.append((str(rel_path), fixes))

    print()
    print(f"=== Summary ===")
    print(f"Files scanned:      {len(c_files)}")
    print(f"Files with issues:  {files_with_fixes}")
    print(f"Total fallthroughs: {total_fixes}")

    if args.dry_run:
        print()
        print("Run without --dry-run to apply fixes.")
    else:
        print()
        print(f"Applied {total_fixes} break; insertions across {files_with_fixes} files.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
