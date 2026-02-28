#!/usr/bin/env python3
"""
fix_switch_fallthrough.py - Find and fix unintentional switch fallthrough in C files.

Scans all .c files under src/cognitive/ (or a specified path) for switch case labels
where the preceding case body has executable statements but does NOT end with
break/return/goto/continue or a fallthrough comment.

CONSERVATIVE approach:
  - Only adds `break;` where it is CLEARLY a bug
  - Skips stacked cases (case A: case B: with no code between)
  - Skips cases terminated by return/goto/continue/break (even multi-line)
  - Skips fallthrough comments (/* fallthrough */, /* FALLTHROUGH */, etc.)
  - Skips default: labels (often intentional)
  - Skips cases where previous case body ends with `}` on its own line
    (often closing an if/for/while block with all paths covered)
  - Handles multi-line return/function-call statements correctly
  - Handles `func(); break;` on a single line

Usage:
    python3 scripts/fix_switch_fallthrough.py --dry-run           # Count only
    python3 scripts/fix_switch_fallthrough.py                     # Apply fixes
    python3 scripts/fix_switch_fallthrough.py --verbose           # Show each fix
    python3 scripts/fix_switch_fallthrough.py --stats             # Show statistics
    python3 scripts/fix_switch_fallthrough.py --path src/         # Scan all of src/
    python3 scripts/fix_switch_fallthrough.py --self-test         # Run self-tests
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Pattern for case/default labels
CASE_LABEL_RE = re.compile(r'^\s*case\s+[A-Za-z_0-9]+\s*:')
DEFAULT_LABEL_RE = re.compile(r'^\s*default\s*:')

# Pattern for fallthrough comments (various common styles)
FALLTHROUGH_COMMENT_RE = re.compile(
    r'/\*\s*(?:fall\s*through|FALL\s*THROUGH|FALLTHROUGH|falls?\s*thro?u?gh)\s*\*/'
    r'|//\s*(?:fall\s*through|FALL\s*THROUGH|FALLTHROUGH|falls?\s*thro?u?gh)',
    re.IGNORECASE
)

# Pattern for opening of a switch statement
SWITCH_OPEN_RE = re.compile(r'^\s*switch\s*\(')

# Flow control keywords that terminate a case body
FLOW_CONTROL_KEYWORDS = {'break', 'return', 'goto', 'continue', 'exit', '_exit', 'abort'}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
    result = re.sub(r'//.*$', '', line)
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
    if stripped.startswith('*') and not stripped.startswith('*/'):
        return True
    if stripped == '*/':
        return True
    return False


def find_statement_block(lines, end_idx):
    """
    Starting from end_idx, walk backwards to reconstruct the full logical
    statement that ends at this line.

    Returns (start_idx, end_idx, concatenated_text).

    Handles multi-line statements like:
        return foo(
            bar, baz,
            qux);
    """
    if end_idx < 0 or end_idx >= len(lines):
        return end_idx, end_idx, ''

    stmt_lines = []
    end_stripped = strip_comments(lines[end_idx]).strip()
    stmt_lines.append(end_stripped)

    # Track unbalanced parentheses going backwards
    paren_depth = end_stripped.count(')') - end_stripped.count('(')

    idx = end_idx - 1
    while idx >= 0:
        stripped = strip_comments(lines[idx]).strip()

        # Stop at boundaries
        if is_blank_or_comment(lines[idx]):
            break
        if is_case_or_default(lines[idx]):
            break
        if stripped.startswith('#'):
            break
        if stripped == '{' or stripped == '}':
            break

        # Track parentheses
        paren_depth += stripped.count(')') - stripped.count('(')

        is_continuation = False

        # Inside unclosed parentheses => continuation
        if paren_depth > 0:
            is_continuation = True
        # Previous line ends with continuation character (binary/unary operator,
        # comma, open paren, backslash, ternary operator)
        elif stripped.endswith((',', '|', '&', '+', '-', '*', '/', '%', '^',
                                '\\', '(', '?', '=', '~')):
            is_continuation = True
        elif stripped.endswith(':') and not is_case_or_default(lines[idx]):
            is_continuation = True
        # Line ends with && or ||
        elif stripped.endswith('&&') or stripped.endswith('||'):
            is_continuation = True

        if is_continuation:
            stmt_lines.append(stripped)
            idx -= 1
        else:
            # Check if this line starts with a flow control keyword
            # that could span to end_idx (multi-line return/goto/etc.)
            found_kw = False
            for kw in FLOW_CONTROL_KEYWORDS:
                if stripped.startswith(kw + ' ') or stripped.startswith(kw + '(') or \
                   stripped.startswith(kw + ';') or stripped == kw:
                    stmt_lines.append(stripped)
                    idx -= 1
                    paren_depth = 0
                    found_kw = True
                    break
            if not found_kw:
                break

    stmt_lines.reverse()
    full_stmt = ' '.join(stmt_lines)
    start_idx = idx + 1

    return start_idx, end_idx, full_stmt


def statement_has_flow_control(stmt_text, last_line_text):
    """
    Check if a statement (possibly multi-line, concatenated) ends execution
    flow for the case body.
    """
    stripped = stmt_text.strip()

    # Check statement start for flow control keywords
    for kw in ('break', 'return', 'goto', 'continue', 'exit', '_exit', 'abort'):
        if stripped.startswith(kw + ' ') or stripped.startswith(kw + '(') or \
           stripped.startswith(kw + ';') or stripped == kw + ';' or stripped == kw:
            return True

    # Check the raw last line for flow control anywhere on it
    # Handles patterns like `func(); break;` or `x = 1; return x;`
    last_stripped = strip_comments(last_line_text).strip()
    if re.search(r'\bbreak\s*;\s*$', last_stripped):
        return True
    if re.search(r'\breturn\b[^;]*;\s*$', last_stripped):
        return True
    if re.search(r'\bgoto\s+\w+\s*;\s*$', last_stripped):
        return True
    if re.search(r'\bcontinue\s*;\s*$', last_stripped):
        return True

    # NIMCP_THROW_TO_IMMUNE often serves as quasi-flow-control
    if 'NIMCP_THROW_TO_IMMUNE' in stripped:
        return True

    return False


def find_preceding_significant_line(lines, case_line_idx):
    """
    Walk backwards from case_line_idx to find the last significant
    (non-blank, non-comment) line before this case label.

    Returns (line_index, line_text) or (None, None).
    """
    idx = case_line_idx - 1
    in_block_comment = False

    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()

        # Track block comments walking backwards
        if stripped.endswith('*/') and not stripped.startswith('//'):
            if '/*' in stripped and not stripped.startswith('/*'):
                # Like: code /* comment */
                return idx, line
            elif stripped.startswith('/*'):
                # Single-line block comment /* ... */
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

        if stripped == '' or stripped.startswith('//'):
            idx -= 1
            continue

        if stripped.startswith('*') and not stripped.startswith('*/'):
            idx -= 1
            continue

        return idx, line

    return None, None


# ---------------------------------------------------------------------------
# File analysis
# ---------------------------------------------------------------------------

def analyze_file(filepath, dry_run=True, verbose=False, collect_stats=False):
    """
    Analyze a single C file for unintentional switch fallthrough.

    Returns (fixes_list, stats_dict).
    fixes_list: list of (line_number, description) for each fix.
    stats_dict: counts for statistics if collect_stats is True.
    """
    stats = {
        'case_labels': 0,
        'stacked_cases': 0,
        'flow_controlled': 0,
        'brace_terminated': 0,
        'fallthrough_commented': 0,
        'unintentional': 0,
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        if verbose:
            print(f"  WARNING: Could not read {filepath}: {e}", file=sys.stderr)
        return [], stats

    fixes = []
    insertions = []

    for i, line in enumerate(lines):
        if not is_case_label(line):
            continue

        stats['case_labels'] += 1

        # Find the preceding significant line
        prev_idx, prev_line = find_preceding_significant_line(lines, i)
        if prev_idx is None:
            continue

        prev_stripped = prev_line.strip()

        # SKIP: Stacked cases (case A: case B: with no code between)
        if is_case_or_default(prev_line):
            stats['stacked_cases'] += 1
            continue

        # SKIP: Opening brace of switch or block
        if prev_stripped == '{':
            continue

        # SKIP: Closing brace (if/for/while block boundary)
        if prev_stripped.startswith('}'):
            stats['brace_terminated'] += 1
            continue

        # SKIP: Preprocessor directive
        if prev_stripped.startswith('#'):
            continue

        # Check for fallthrough comments
        has_ft_between = False
        for j in range(prev_idx, i):
            if has_fallthrough_comment(lines[j]):
                has_ft_between = True
                break
        if has_ft_between:
            stats['fallthrough_commented'] += 1
            continue

        # Reconstruct multi-line statement
        stmt_start, stmt_end, full_stmt = find_statement_block(lines, prev_idx)

        # Check for flow control
        if statement_has_flow_control(full_stmt, prev_line):
            stats['flow_controlled'] += 1
            continue

        # Verify this looks like a real statement (ends with ; or ))
        code_stripped = strip_comments(prev_line).strip()
        if not (code_stripped.endswith(';') or code_stripped.endswith(')') or
                code_stripped.endswith(',')):
            continue

        # Verify there is an owning case/default label above
        found_owning_case = False
        brace_depth = 0
        check_idx = stmt_start
        while check_idx >= 0:
            check_stripped = lines[check_idx].strip()
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

        # This is an unintentional fallthrough
        stats['unintentional'] += 1

        indent = get_indentation(prev_line)
        line_num = i + 1
        prev_line_num = prev_idx + 1
        desc = f"Line {line_num}: missing break; after line {prev_line_num}: {prev_stripped[:80]}"
        fixes.append((line_num, desc))
        insertions.append((prev_idx + 1, indent, desc))

    # Apply fixes if not dry-run
    if not dry_run and insertions:
        insertions.sort(key=lambda x: x[0], reverse=True)
        for insert_idx, indent, desc in insertions:
            break_line = f"{indent}break;\n"
            lines.insert(insert_idx, break_line)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except (IOError, OSError) as e:
            print(f"  ERROR: Could not write {filepath}: {e}", file=sys.stderr)
            return [], stats

    return fixes, stats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_tests():
    """Run self-tests with synthetic C code to verify detection logic."""
    test_cases = [
        # Test 1: Clear unintentional fallthrough
        (
            "unintentional_simple",
            """\
switch (x) {
    case FOO:
        x = 1;
    case BAR:
        x = 2;
        break;
}
""",
            1,  # Expected: 1 fix
        ),

        # Test 2: Proper break — no fix needed
        (
            "proper_break",
            """\
switch (x) {
    case FOO:
        x = 1;
        break;
    case BAR:
        x = 2;
        break;
}
""",
            0,
        ),

        # Test 3: Multi-line return — no fix needed
        (
            "multiline_return",
            """\
switch (x) {
    case FOO:
        return compute(
            a, b,
            c);
    case BAR:
        return 0;
}
""",
            0,
        ),

        # Test 4: Stacked cases — no fix needed
        (
            "stacked_cases",
            """\
switch (x) {
    case FOO:
    case BAR:
        x = 1;
        break;
    case BAZ:
        x = 2;
        break;
}
""",
            0,
        ),

        # Test 5: func(); break; on same line — no fix needed
        (
            "inline_break",
            """\
switch (x) {
    case FOO:
        do_something(); break;
    case BAR:
        do_other(); break;
}
""",
            0,
        ),

        # Test 6: Closing brace — no fix (conservative)
        (
            "closing_brace",
            """\
switch (x) {
    case FOO:
        if (y) {
            return 1;
        }
    case BAR:
        break;
}
""",
            0,
        ),

        # Test 7: Fallthrough comment — no fix needed
        (
            "fallthrough_comment",
            """\
switch (x) {
    case FOO:
        x = 1;
        /* fallthrough */
    case BAR:
        x = 2;
        break;
}
""",
            0,
        ),

        # Test 8: default label — skipped
        (
            "default_label",
            """\
switch (x) {
    case FOO:
        x = 1;
    default:
        x = 2;
        break;
}
""",
            0,  # We skip default: labels
        ),

        # Test 9: Multi-line return with operators
        (
            "multiline_return_ops",
            """\
switch (provider) {
    case PROVIDER_A:
        return CAP_TEXT | CAP_IMAGE |
               CAP_VIDEO | CAP_AUDIO;
    case PROVIDER_B:
        return CAP_TEXT;
}
""",
            0,
        ),

        # Test 10: Multiple unintentional fallthroughs
        (
            "multiple_unintentional",
            """\
switch (x) {
    case A:
        x = 1;
    case B:
        y = 2;
    case C:
        z = 3;
        break;
}
""",
            2,  # A->B and B->C
        ),

        # Test 11: NIMCP_THROW_TO_IMMUNE — treated as flow control
        (
            "nimcp_throw",
            """\
switch (x) {
    case FOO:
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "msg");
    case BAR:
        break;
}
""",
            0,
        ),
    ]

    passed = 0
    failed = 0

    for name, code, expected_fixes in test_cases:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(code)
            tmppath = f.name

        try:
            fixes, stats = analyze_file(tmppath, dry_run=True, verbose=False,
                                         collect_stats=True)
            actual = len(fixes)

            if actual == expected_fixes:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
                print(f"  {status}: {name} - expected {expected_fixes}, got {actual}")
                for _, desc in fixes:
                    print(f"    {desc}")
                if expected_fixes > 0 and actual == 0:
                    print(f"    (no fixes found but {expected_fixes} expected)")
        finally:
            os.unlink(tmppath)

    print(f"\nSelf-test results: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Find and fix unintentional switch fallthrough in C files'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report, do not modify files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show each fix location')
    parser.add_argument('--stats', action='store_true',
                        help='Show per-file statistics breakdown')
    parser.add_argument('--path', default=None,
                        help='Override scan path (default: src/cognitive/)')
    parser.add_argument('--self-test', action='store_true',
                        help='Run self-tests and exit')
    args = parser.parse_args()

    if args.self_test:
        print("=== Running self-tests ===\n")
        success = run_self_tests()
        return 0 if success else 1

    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.path:
        scan_dir = Path(args.path)
        if not scan_dir.is_absolute():
            scan_dir = project_root / scan_dir
    else:
        scan_dir = project_root / 'src' / 'cognitive'

    if not scan_dir.is_dir():
        print(f"ERROR: Directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "APPLYING FIXES"
    print(f"=== Switch Fallthrough Fixer ({mode}) ===")
    print(f"Scanning: {scan_dir}")
    print()

    c_files = sorted(scan_dir.rglob('*.c'))
    print(f"Found {len(c_files)} .c files to scan")
    print()

    total_fixes = 0
    files_with_fixes = 0
    all_fixes = []

    # Aggregate stats
    agg_stats = {
        'case_labels': 0,
        'stacked_cases': 0,
        'flow_controlled': 0,
        'brace_terminated': 0,
        'fallthrough_commented': 0,
        'unintentional': 0,
    }

    for filepath in c_files:
        fixes, stats = analyze_file(
            str(filepath), dry_run=args.dry_run,
            verbose=args.verbose, collect_stats=args.stats
        )

        # Accumulate stats
        for key in agg_stats:
            agg_stats[key] += stats[key]

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
    print("=== Summary ===")
    print(f"Files scanned:      {len(c_files)}")
    print(f"Files with issues:  {files_with_fixes}")
    print(f"Total fallthroughs: {total_fixes}")

    if args.stats:
        print()
        print("=== Case Label Statistics ===")
        print(f"Total case labels examined:     {agg_stats['case_labels']}")
        print(f"  Stacked (intentional):        {agg_stats['stacked_cases']}")
        print(f"  Flow-control terminated:      {agg_stats['flow_controlled']}")
        print(f"  Brace-terminated (skipped):   {agg_stats['brace_terminated']}")
        print(f"  Fallthrough-commented:        {agg_stats['fallthrough_commented']}")
        print(f"  Unintentional (needs break):  {agg_stats['unintentional']}")

    if args.dry_run and total_fixes > 0:
        print()
        print("Run without --dry-run to apply fixes.")
    elif not args.dry_run and total_fixes > 0:
        print()
        print(f"Applied {total_fixes} break; insertions across {files_with_fixes} files.")
    elif total_fixes == 0:
        print()
        print("No unintentional fallthroughs detected. Code is clean.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
