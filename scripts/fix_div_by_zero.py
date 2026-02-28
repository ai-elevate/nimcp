#!/usr/bin/env python3
"""
Fix division-by-zero bugs in the cognitive layer.

Scans all .c files in src/cognitive/ and wraps unguarded divisor variables
with safe checks to prevent division by zero.

Usage:
    python3 scripts/fix_div_by_zero.py --dry-run   # Count fixes without modifying
    python3 scripts/fix_div_by_zero.py              # Apply fixes
"""

import os
import re
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COGNITIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "src", "cognitive")

# Variable names that plausibly could be zero
RISKY_NAMES = {
    "count", "n", "num", "size", "len", "sum", "norm", "total", "dt",
    "denominator", "divisor", "weight", "magnitude", "distance", "area",
    "volume", "scale", "ratio", "avg", "mean", "std", "variance", "sigma",
    # Common suffixed forms
    "num_items", "num_elements", "num_steps", "num_samples", "num_neurons",
    "num_layers", "num_entries", "num_inputs", "num_outputs", "num_hidden",
    "num_weights", "num_patterns", "num_rules", "num_nodes", "num_edges",
    "num_classes", "num_agents", "num_strategies", "num_actions",
    "num_categories", "num_features", "num_concepts", "num_values",
    "num_components", "num_active", "num_valid", "num_responses",
    "num_candidates", "num_hypotheses", "num_chains", "num_steps_completed",
    "total_count", "total_weight", "total_neurons", "total_synapses",
    "total_time", "total_updates", "total_distance",
    "step_count", "update_count", "sample_count", "match_count",
    "active_count", "valid_count", "success_count", "error_count",
    "hit_count", "miss_count", "eval_count", "batch_count",
    "neuron_count", "synapse_count", "layer_count", "node_count",
    "edge_count", "rule_count", "pattern_count", "concept_count",
    "cluster_count", "group_count", "member_count", "candidate_count",
    "level_count", "chain_count", "action_count", "response_count",
    "observation_count", "prediction_count", "episode_count",
    "window_size", "batch_size", "block_size", "layer_size",
    "input_size", "output_size", "hidden_size", "embed_size",
    "vocab_size", "max_size", "buf_size", "buffer_size",
    "input_dim", "output_dim", "hidden_dim", "embed_dim",
    "total_sum", "weight_sum", "prob_sum", "score_sum",
    "norm_val", "norm_sq", "norm_factor",
    "stddev", "std_dev",
    "time_span", "time_delta", "elapsed", "duration",
    "inv_norm", "inv_std", "inv_variance",
}

# Also match names ending with these suffixes via pattern
RISKY_SUFFIXES = (
    "_count", "_num", "_size", "_len", "_sum", "_norm", "_total",
    "_weight", "_magnitude", "_distance", "_scale", "_ratio",
    "_avg", "_mean", "_std", "_variance", "_sigma",
    "_denominator", "_divisor", "_area", "_volume", "_dt",
)

# Integer type keywords (used to decide guard style)
INTEGER_TYPE_KEYWORDS = {
    "int", "uint32_t", "uint64_t", "int32_t", "int64_t", "size_t",
    "uint16_t", "int16_t", "uint8_t", "int8_t", "unsigned", "long",
    "ssize_t", "ptrdiff_t",
}

# Float type keywords
FLOAT_TYPE_KEYWORDS = {
    "float", "double", "nimcp_real_t",
}


def is_risky_name(name: str) -> bool:
    """Check if a variable name suggests it could plausibly be zero."""
    lower = name.lower()
    if lower in RISKY_NAMES:
        return True
    for suffix in RISKY_SUFFIXES:
        if lower.endswith(suffix):
            return True
    return False


def is_inside_string(line: str, match_start: int) -> bool:
    """Heuristic check: is the match position inside a string literal?"""
    in_string = False
    quote_char = None
    i = 0
    while i < match_start:
        ch = line[i]
        if ch == '\\' and in_string:
            i += 2  # skip escaped char
            continue
        if ch in ('"', "'"):
            if not in_string:
                in_string = True
                quote_char = ch
            elif ch == quote_char:
                in_string = False
        i += 1
    return in_string


def is_comment_line(line: str) -> bool:
    """Check if the line is a comment (// or starts with *)."""
    stripped = line.lstrip()
    if stripped.startswith("//"):
        return True
    if stripped.startswith("/*"):
        return True
    if stripped.startswith("*"):
        return True
    return False


def is_inside_line_comment(line: str, pos: int) -> bool:
    """Check if position is after a // comment start."""
    comment_pos = line.find("//")
    if comment_pos >= 0 and comment_pos < pos:
        return True
    return False


def is_already_guarded(lines: list, line_idx: int, var_name: str) -> bool:
    """
    Check if the division is already guarded by a zero check.
    Looks at the current line and surrounding context for patterns like:
      - if (var == 0) return ...
      - if (var > 0) { ... / var ... }
      - (var > 0) ? (... / var) : ...
      - var > 0 ? ... / var : ...
      - fabsf(var) > ... ? var : ...
    Also traces back through enclosing brace blocks to find guards.
    """
    line = lines[line_idx]
    esc = re.escape(var_name)

    # Check current line for ternary guard or inline guard
    ternary_patterns = [
        rf'{esc}\s*>\s*0\s*\?',            # var > 0 ?
        rf'{esc}\s*!=\s*0\s*\?',           # var != 0 ?
        rf'{esc}\s*>\s*[\d.]+[fF]?\s*\?',  # var > 0.0f ?
        rf'fabsf?\s*\(\s*{esc}\s*\)\s*>',  # fabsf(var) >
        rf'{esc}\s*>\s*0\s*\?\s*.*/',      # var > 0 ? .../
        rf'\(\s*{esc}\s*>\s*0',             # (var > 0
        rf'\(\s*{esc}\s*!=\s*0',            # (var != 0
        rf'if\s*\(\s*{esc}\s*>\s*0',        # if (var > 0 ... on same line
        rf'if\s*\(\s*{esc}\s*!=\s*0',       # if (var != 0 ... on same line
    ]
    for pat in ternary_patterns:
        if re.search(pat, line):
            return True

    # Guard patterns for preceding lines
    guard_patterns = [
        rf'if\s*\(\s*{esc}\s*==\s*0',
        rf'if\s*\(\s*{esc}\s*<=?\s*0',
        rf'if\s*\(\s*{esc}\s*<\s*1\b',
        rf'if\s*\(\s*!{esc}\b',
        rf'if\s*\(\s*{esc}\s*>\s*0',
        rf'if\s*\(\s*{esc}\s*!=\s*0',
        rf'{esc}\s*=\s*.*\?\s*.*:\s*1',  # var = ... ? ... : 1
    ]

    # Strategy 1: Simple lookback (up to 5 lines) for direct guards
    start = max(0, line_idx - 5)
    for i in range(start, line_idx):
        prev = lines[i]
        for pat in guard_patterns:
            if re.search(pat, prev):
                return True

    # Strategy 2: Trace back through enclosing { } blocks
    # Walk backwards from current line, tracking brace nesting.
    # If we find an opening { whose preceding line has a guard for var_name,
    # then the division is inside a guarded block.
    brace_depth = 0
    scan_start = max(0, line_idx - 60)  # Don't look too far back
    for i in range(line_idx, scan_start - 1, -1):
        ln = lines[i]
        # Count braces (simplified: ignoring strings/comments for speed)
        for ch in reversed(ln):
            if ch == '}':
                brace_depth += 1
            elif ch == '{':
                brace_depth -= 1

        # When brace_depth goes negative, we found an enclosing {
        if brace_depth < 0:
            # Check the line with the { and the line before it for a guard
            for check_idx in range(max(0, i - 1), min(len(lines), i + 1)):
                check_line = lines[check_idx]
                for pat in guard_patterns:
                    if re.search(pat, check_line):
                        return True
            # Reset and keep scanning for outer blocks
            brace_depth = 0

    return False


def guess_type_context(lines: list, line_idx: int, var_name: str) -> str:
    """
    Try to determine if the variable is integer or float type.
    Returns 'int', 'float', or 'unknown'.
    """
    # Search backwards for a declaration or cast of this variable
    start = max(0, line_idx - 30)
    for i in range(line_idx, start - 1, -1):
        ln = lines[i]
        # Look for type declaration: "type var_name" or "type var_name ="
        for kw in FLOAT_TYPE_KEYWORDS:
            if re.search(rf'\b{kw}\b\s+\*?\s*{re.escape(var_name)}\b', ln):
                return 'float'
        for kw in INTEGER_TYPE_KEYWORDS:
            if re.search(rf'\b{kw}\b\s+\*?\s*{re.escape(var_name)}\b', ln):
                return 'int'
        # Check for cast: (float)var_name or (uint32_t)var_name
        if re.search(rf'\(\s*(?:float|double)\s*\)\s*{re.escape(var_name)}\b', ln):
            return 'float'

    # Heuristic: names like dt, norm, sigma, std, mean, avg, variance, weight,
    # distance, scale, ratio are almost always float
    float_hints = {
        "dt", "norm", "sigma", "std", "stddev", "std_dev", "mean", "avg",
        "variance", "weight", "distance", "scale", "ratio", "magnitude",
        "sum", "total_weight", "weight_sum", "prob_sum", "score_sum",
        "norm_val", "norm_sq", "norm_factor", "inv_norm", "inv_std",
        "inv_variance", "time_span", "time_delta", "elapsed", "duration",
        "denominator", "area", "volume",
    }
    int_hints = {
        "count", "n", "num", "size", "len", "total", "total_count",
        "step_count", "update_count", "sample_count", "batch_count",
        "neuron_count", "synapse_count", "layer_count", "node_count",
        "edge_count", "match_count", "active_count", "valid_count",
        "success_count", "error_count", "hit_count", "miss_count",
        "eval_count", "batch_count", "rule_count", "pattern_count",
        "concept_count", "cluster_count", "group_count", "member_count",
        "candidate_count", "level_count", "chain_count", "action_count",
        "response_count", "observation_count", "prediction_count",
        "episode_count", "num_items", "num_elements", "num_steps",
        "num_samples", "num_neurons", "num_layers", "num_entries",
        "num_inputs", "num_outputs", "num_hidden", "num_weights",
        "num_patterns", "num_rules", "num_nodes", "num_edges",
        "num_classes", "num_agents", "num_strategies", "num_actions",
        "num_categories", "num_features", "num_concepts", "num_values",
        "num_components", "num_active", "num_valid", "num_responses",
        "num_candidates", "num_hypotheses", "num_chains",
        "window_size", "batch_size", "block_size", "layer_size",
        "input_size", "output_size", "hidden_size", "embed_size",
        "vocab_size", "max_size", "buf_size", "buffer_size",
        "input_dim", "output_dim", "hidden_dim", "embed_dim",
    }
    lower = var_name.lower()
    if lower in float_hints:
        return 'float'
    if lower in int_hints:
        return 'int'
    # Suffix heuristics
    for suffix in ("_count", "_num", "_size", "_len", "_dim"):
        if lower.endswith(suffix):
            return 'int'
    for suffix in ("_norm", "_weight", "_sum", "_mean", "_avg", "_std",
                    "_variance", "_sigma", "_scale", "_ratio", "_distance"):
        if lower.endswith(suffix):
            return 'float'

    return 'unknown'


def make_safe_divisor(var_name: str, var_type: str) -> str:
    """Generate a safe divisor expression."""
    if var_type == 'int':
        return f"({var_name} > 0 ? {var_name} : 1)"
    elif var_type == 'float':
        return f"(fabsf({var_name}) > 1e-7f ? {var_name} : 1e-7f)"
    else:
        # Default: treat as float (safer for unknown types since most
        # divisions in this codebase are floating point)
        return f"(fabsf({var_name}) > 1e-7f ? {var_name} : 1e-7f)"


# Regex to find division by a simple variable name
# Matches: / var_name  where var_name is a C identifier
# Captures: (slash)(optional space)(variable name)
# Does NOT match: /= , /* (comment), // (comment), /-> , /.member
DIV_PATTERN = re.compile(
    r'(?<![/\w])'          # Not preceded by / or word char (avoid // and member/)
    r'(/)\s*'              # The division operator
    r'([a-zA-Z_]\w*)'     # The variable name
    r'(?!\s*[(/\w.])'     # Not followed by ( [function call], / [comment], word char, . [member]
)

# More precise pattern: match " / var" making sure it's a division context
# We look for: <something> / var where <something> suggests arithmetic
DIVISION_PATTERN = re.compile(
    r'(\w[\w\[\]\.\->]*\s*)'  # LHS of division (variable, member, array access)
    r'(/)\s*'                   # Division operator
    r'([a-zA-Z_]\w*)'          # Divisor variable name
    r'\b'                       # Word boundary
    r'(?!\s*\()'               # Not followed by ( (would be function call)
)

# Even simpler: just find all " / identifier" patterns
SIMPLE_DIV = re.compile(
    r'(?<=[\w\)\]\s])'   # Preceded by word char, ), ], or space (left operand)
    r'\s*/\s*'            # Division operator with optional whitespace
    r'([a-zA-Z_]\w*)'    # Divisor variable name
    r'\b'                 # Word boundary
    r'(?!\s*\()'          # Not followed by ( (not a function call like / func())
)


def find_divisions_in_line(line: str) -> list:
    """
    Find all division-by-variable instances in a line.
    Returns list of (match_start, match_end, var_name, full_match_text).
    """
    results = []
    # Use a more careful approach: find all '/' that look like division
    i = 0
    while i < len(line):
        # Find next '/'
        pos = line.find('/', i)
        if pos < 0:
            break

        # Skip // comments
        if pos + 1 < len(line) and line[pos + 1] == '/':
            break  # Rest of line is comment
        # Skip /* comments
        if pos + 1 < len(line) and line[pos + 1] == '*':
            # Find end of block comment on this line
            end_comment = line.find('*/', pos + 2)
            if end_comment >= 0:
                i = end_comment + 2
                continue
            else:
                break  # Rest of line is in block comment
        # Skip /= (compound assignment — we could fix these too, but skip for safety)
        if pos + 1 < len(line) and line[pos + 1] == '=':
            i = pos + 2
            continue

        # Check that this looks like a division operator:
        # Should have something before it (not start of significant expression)
        before = line[:pos].rstrip()
        if not before or before[-1] not in (')', ']', '0', '1', '2', '3', '4', '5',
                                              '6', '7', '8', '9', 'f', 'F', 'u', 'U',
                                              'l', 'L') and \
           not re.search(r'[a-zA-Z_]\w*$', before):
            i = pos + 1
            continue

        # Now look at what follows the /
        after = line[pos + 1:].lstrip()
        # Match a simple variable name
        m = re.match(r'([a-zA-Z_]\w*)\b', after)
        if not m:
            i = pos + 1
            continue

        var_name = m.group(1)

        # Check it's not a keyword or type
        c_keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
            'continue', 'return', 'goto', 'sizeof', 'typeof', 'void', 'char',
            'short', 'int', 'long', 'float', 'double', 'unsigned', 'signed',
            'struct', 'union', 'enum', 'typedef', 'const', 'volatile',
            'static', 'extern', 'register', 'auto', 'inline', 'NULL',
            'true', 'false', 'bool', 'uint8_t', 'uint16_t', 'uint32_t',
            'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'size_t', 'ssize_t', 'ptrdiff_t',
        }
        if var_name in c_keywords:
            i = pos + 1
            continue

        # Check the character after the variable name
        after_var = after[len(var_name):]
        # If followed by ( it's a function call, skip
        if after_var.lstrip().startswith('('):
            i = pos + 1
            continue
        # If followed by -> or . it's a member access - the divisor is a complex
        # expression, skip (we only fix simple variable names)
        if after_var.lstrip().startswith('->') or after_var.lstrip().startswith('.'):
            i = pos + 1
            continue
        # If followed by [ it's an array subscript - the divisor is var[index],
        # not a simple variable. Skip (wrapping would break syntax).
        if after_var.lstrip().startswith('['):
            i = pos + 1
            continue

        # Calculate the actual position in the line of the var_name
        # The / is at pos, then there might be whitespace
        space_after_slash = len(line[pos + 1:]) - len(line[pos + 1:].lstrip())
        var_start = pos + 1 + space_after_slash
        var_end = var_start + len(var_name)

        results.append((pos, var_end, var_name))
        i = var_end

    return results


def process_file(filepath: str, dry_run: bool = False) -> int:
    """Process a single .c file. Returns count of fixes made."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  WARNING: Could not read {filepath}: {e}", file=sys.stderr)
        return 0

    fixes = 0
    new_lines = []
    in_block_comment = False

    for line_idx, line in enumerate(lines):
        # Track block comments
        if in_block_comment:
            if '*/' in line:
                in_block_comment = False
            new_lines.append(line)
            continue

        if '/*' in line and '*/' not in line:
            # Check if /* is not inside a string
            comment_pos = line.find('/*')
            if not is_inside_string(line, comment_pos):
                in_block_comment = True
                new_lines.append(line)
                continue

        # Skip pure comment lines
        if is_comment_line(line):
            new_lines.append(line)
            continue

        # Skip preprocessor directives
        stripped = line.lstrip()
        if stripped.startswith('#'):
            new_lines.append(line)
            continue

        # Find divisions in this line
        divisions = find_divisions_in_line(line)
        if not divisions:
            new_lines.append(line)
            continue

        modified_line = line
        offset = 0  # Track offset from insertions

        for slash_pos, var_end, var_name in divisions:
            # Check if the variable name is risky
            if not is_risky_name(var_name):
                continue

            # Check if inside string literal
            if is_inside_string(line, slash_pos):
                continue

            # Check if after // comment
            if is_inside_line_comment(line, slash_pos):
                continue

            # Check if already guarded
            if is_already_guarded(lines, line_idx, var_name):
                continue

            # Determine the type context
            var_type = guess_type_context(lines, line_idx, var_name)

            # Generate the safe replacement
            safe_divisor = make_safe_divisor(var_name, var_type)

            # Replace: find the exact var_name occurrence at the right position
            # We need to replace just the var_name after the / with the safe expression
            actual_slash_pos = slash_pos + offset
            actual_var_end = var_end + offset

            # Find the var_name in the modified_line at the expected position
            # The region from actual_slash_pos to actual_var_end should contain "/ var_name"
            region = modified_line[actual_slash_pos:actual_var_end]
            # Double-check it contains our variable
            if var_name in region:
                # Replace just the variable name portion
                var_start_in_region = region.rfind(var_name)
                abs_var_start = actual_slash_pos + var_start_in_region
                abs_var_end = abs_var_start + len(var_name)

                modified_line = (modified_line[:abs_var_start] +
                                safe_divisor +
                                modified_line[abs_var_end:])
                offset += len(safe_divisor) - len(var_name)
                fixes += 1

        new_lines.append(modified_line)

    if fixes > 0 and not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"  WARNING: Could not write {filepath}: {e}", file=sys.stderr)
            return 0

    return fixes


def check_needs_fabsf_include(filepath: str) -> bool:
    """Check if a file uses fabsf but doesn't include math.h."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return False

    if 'fabsf(' not in content:
        return False
    if '#include <math.h>' in content or '#include "nimcp_math' in content:
        return False
    return True


def add_math_include_if_needed(filepath: str, dry_run: bool = False) -> bool:
    """Add #include <math.h> if the file uses fabsf but doesn't have it."""
    if not check_needs_fabsf_include(filepath):
        return False

    if dry_run:
        return True

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return False

    # Find the last #include line and insert after it
    last_include = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            last_include = i

    if last_include >= 0:
        lines.insert(last_include + 1, '#include <math.h>\n')
    else:
        # No includes found, add at beginning
        lines.insert(0, '#include <math.h>\n')

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix division-by-zero bugs in the cognitive layer"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Count fixes without modifying files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each fix")
    args = parser.parse_args()

    if not os.path.isdir(COGNITIVE_DIR):
        print(f"ERROR: Directory not found: {COGNITIVE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning .c files in: {COGNITIVE_DIR}")
    if args.dry_run:
        print("Mode: DRY RUN (no files will be modified)")
    else:
        print("Mode: APPLYING FIXES")
    print()

    # Collect all .c files
    c_files = []
    for root, dirs, files in os.walk(COGNITIVE_DIR):
        for fname in sorted(files):
            if fname.endswith('.c'):
                c_files.append(os.path.join(root, fname))

    c_files.sort()
    print(f"Found {len(c_files)} .c files")
    print()

    total_fixes = 0
    files_fixed = 0
    math_includes_added = 0

    for filepath in c_files:
        rel_path = os.path.relpath(filepath, os.path.dirname(COGNITIVE_DIR))
        fix_count = process_file(filepath, dry_run=args.dry_run)

        if fix_count > 0:
            files_fixed += 1
            total_fixes += fix_count
            if args.verbose or args.dry_run:
                print(f"  {rel_path}: {fix_count} fix(es)")

            # Check if we need to add math.h for fabsf
            if add_math_include_if_needed(filepath, dry_run=args.dry_run):
                math_includes_added += 1
                if args.verbose or args.dry_run:
                    print(f"    + added #include <math.h>")

    print()
    print("=" * 60)
    print(f"Total division-by-zero fixes: {total_fixes}")
    print(f"Files modified: {files_fixed}")
    print(f"math.h includes added: {math_includes_added}")
    if args.dry_run:
        print("(DRY RUN - no files were modified)")
    print("=" * 60)


if __name__ == "__main__":
    main()
