#!/usr/bin/env python3
"""
Replace magic math constants with named constants from nimcp_math_constants.h.

Three replacement categories:
  1. Local #define M_PI/TWO_PI/SQRT2/etc. -> remove, add include
  2. Inline 3.14159f literals -> NIMCP_PI_F
  3. Inline 6.2831853f / 1.41421356f / 2.71828f literals -> named constants

Usage: python3 scripts/replace_magic_math.py [--dry-run]
"""

import os
import re
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
INCLUDE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")
HEADER = "constants/nimcp_math_constants.h"
DRY_RUN = "--dry-run" in sys.argv

# ---------------------------------------------------------------------------
# Local #define patterns to remove (replaced by the centralized header)
# ---------------------------------------------------------------------------
LOCAL_DEFINE_PATTERNS = [
    # #define M_PI 3.14159...
    re.compile(r'^(\s*)#\s*define\s+M_PI\s+3\.14159\S*\s*$'),
    # #define TWO_PI 6.28318...
    re.compile(r'^(\s*)#\s*define\s+TWO_PI\s+6\.28\S*\s*$'),
    # #define M_2PI 6.28318...
    re.compile(r'^(\s*)#\s*define\s+M_2PI\s+6\.28\S*\s*$'),
    # #define SQRT2 1.41421...
    re.compile(r'^(\s*)#\s*define\s+SQRT2\s+1\.41421\S*\s*$'),
    # #define SQRT_2 1.41421...
    re.compile(r'^(\s*)#\s*define\s+SQRT_2\s+1\.41421\S*\s*$'),
    # #define FIN_MKT_SQRT2 1.41421...
    re.compile(r'^(\s*)#\s*define\s+FIN_MKT_SQRT2\s+1\.41421\S*\s*$'),
]

# Also remove #ifndef M_PI / #endif pairs wrapping the local defines
IFNDEF_MPI_PATTERN = re.compile(r'^(\s*)#\s*ifndef\s+M_PI\s*$')

# ---------------------------------------------------------------------------
# Inline literal replacement rules
# ---------------------------------------------------------------------------
# Each rule: (regex, replacement)
# Order matters: longer/more specific patterns first

INLINE_RULES = [
    # 2.0f * 3.14159f  ->  NIMCP_TWO_PI_F  (common idiom)
    (re.compile(r'2\.0f\s*\*\s*3\.14159\d*f'), 'NIMCP_TWO_PI_F'),

    # 6.2831853f / 6.28318f / 6.283185307179586f  ->  NIMCP_TWO_PI_F
    (re.compile(r'6\.283\d*f'), 'NIMCP_TWO_PI_F'),

    # 6.28318530717958647692 (double, no f suffix)
    (re.compile(r'6\.283185307179586\d*(?!f)(?=[^0-9])'), 'NIMCP_TWO_PI'),

    # 3.14159265358979323846f  ->  NIMCP_PI_F  (full precision float)
    (re.compile(r'3\.14159265358979323846f'), 'NIMCP_PI_F'),

    # 3.14159f / 3.141592f / etc  ->  NIMCP_PI_F  (any truncated pi float)
    (re.compile(r'3\.14159\d*f'), 'NIMCP_PI_F'),

    # 3.14159265358979323846 (double, no f suffix) - used in some #defines
    (re.compile(r'3\.14159265358979323846(?!f)'), 'NIMCP_PI'),

    # 1.41421356237309504880f  ->  NIMCP_SQRT2_F
    (re.compile(r'1\.41421\d*f'), 'NIMCP_SQRT2_F'),

    # 1.41421356237309504880 (double)
    (re.compile(r'1\.414213562373\d*(?!f)(?=[^0-9])'), 'NIMCP_SQRT2'),

    # 2.71828f / 2.71828182845904523536f  ->  NIMCP_EULER_F
    (re.compile(r'2\.71828\d*f'), 'NIMCP_EULER_F'),
]

# ---------------------------------------------------------------------------
# Skip patterns - don't replace inside these
# ---------------------------------------------------------------------------
SKIP_LINE_PATTERNS = [
    re.compile(r'^\s*//'),           # comment-only lines
    re.compile(r'^\s*/\*'),          # block comment starts
    re.compile(r'^\s*\*'),           # block comment continuation
    re.compile(r'NIMCP_PI'),         # already uses our constants
    re.compile(r'NIMCP_TWO_PI'),
    re.compile(r'NIMCP_SQRT2'),
    re.compile(r'NIMCP_EULER'),
    re.compile(r'NIMCP_HALF_PI'),
    re.compile(r'NIMCP_INV_SQRT2'),
]

# Files to skip entirely
SKIP_FILES = {
    "nimcp_math_constants.h",  # our own header
}


def find_include_insertion_point(lines):
    """Find line index after the last #include in the first contiguous include block."""
    first_include = -1
    last_in_block = -1
    in_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include'):
            if first_include == -1:
                first_include = i
                in_block = True
            if in_block:
                last_in_block = i
        elif in_block and stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('#'):
            # Non-include, non-comment, non-preprocessor line -> end of block
            break

    if last_in_block >= 0:
        return last_in_block + 1
    return 0


def already_has_include(lines):
    """Check if file already includes our math constants header."""
    for line in lines:
        if HEADER in line:
            return True
    return False


def process_file(filepath):
    """Process a single file. Returns (defines_removed, literals_replaced, include_added)."""
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return 0, 0, False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return 0, 0, False

    original = list(lines)
    defines_removed = 0
    literals_replaced = 0
    needs_include = False

    # Pass 1: Remove local #define M_PI / TWO_PI / SQRT2 etc.
    # Also handle #ifndef M_PI / #define M_PI ... / #endif triplets
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for #ifndef M_PI ... #define M_PI ... #endif triplet
        if IFNDEF_MPI_PATTERN.match(line):
            # Look ahead for #define M_PI on next line and #endif after
            if i + 2 < len(lines):
                next_line = lines[i + 1]
                endif_line = lines[i + 2]
                is_mpi_define = any(p.match(next_line) for p in LOCAL_DEFINE_PATTERNS)
                is_endif = re.match(r'^\s*#\s*endif', endif_line)
                if is_mpi_define and is_endif:
                    defines_removed += 1
                    needs_include = True
                    i += 3
                    continue

        # Check for standalone local #define
        removed = False
        for pat in LOCAL_DEFINE_PATTERNS:
            if pat.match(line):
                defines_removed += 1
                needs_include = True
                removed = True
                break

        if not removed:
            new_lines.append(line)
        i += 1

    lines = new_lines

    # Track which macro names were removed so we can replace their usage
    removed_macros = set()
    for pat in LOCAL_DEFINE_PATTERNS:
        pat_str = pat.pattern
        # Extract macro name from pattern (between \s+ and \s+)
        if 'TWO_PI' in pat_str:
            removed_macros.add('TWO_PI')
        elif 'M_2PI' in pat_str:
            removed_macros.add('M_2PI')
        elif 'SQRT2' in pat_str and 'SQRT_2' not in pat_str and 'FIN_MKT' not in pat_str:
            removed_macros.add('SQRT2')
        elif 'SQRT_2' in pat_str:
            removed_macros.add('SQRT_2')
        elif 'FIN_MKT_SQRT2' in pat_str:
            removed_macros.add('FIN_MKT_SQRT2')

    # Pass 1b: If we removed a local #define for TWO_PI/SQRT2/etc.,
    # replace all usage of those macro names with NIMCP_* equivalents.
    # (Don't replace M_PI usage - it's provided as a compatibility alias.)
    MACRO_REPLACEMENTS = {
        'TWO_PI': 'NIMCP_TWO_PI_F',
        'M_2PI': 'NIMCP_TWO_PI_F',
        'SQRT2': 'NIMCP_SQRT2_F',
        'SQRT_2': 'NIMCP_SQRT2_F',
        'FIN_MKT_SQRT2': 'NIMCP_SQRT2_F',
    }
    if defines_removed > 0:
        for i, line in enumerate(lines):
            # Skip comment lines and lines that already use NIMCP_ names
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                continue
            if 'NIMCP_' in line:
                continue
            # Skip #define lines (don't replace macro definitions that compute from M_PI)
            if stripped.startswith('#define'):
                continue

            new_line = line
            for macro_name, replacement in MACRO_REPLACEMENTS.items():
                # Only replace macros that were actually removed from THIS file
                # Check if file originally had this specific define
                pat = re.compile(r'\b' + re.escape(macro_name) + r'\b')
                if pat.search(new_line):
                    # Check if the original file had a #define for this macro
                    had_define = False
                    for orig_line in original:
                        if re.match(r'^\s*#\s*define\s+' + re.escape(macro_name) + r'\s', orig_line):
                            had_define = True
                            break
                    if had_define:
                        new_line = pat.sub(replacement, new_line)
                        literals_replaced += 1
                        needs_include = True

            lines[i] = new_line

    # Pass 2: Replace inline literals
    for i, line in enumerate(lines):
        # Skip lines matching skip patterns
        if any(sp.search(line) for sp in SKIP_LINE_PATTERNS):
            continue

        new_line = line
        for regex, replacement in INLINE_RULES:
            if regex.search(new_line):
                new_line = regex.sub(replacement, new_line)
                literals_replaced += 1
                needs_include = True
                break  # one replacement per line to avoid double-counting

        lines[i] = new_line

    # Pass 3: Add include if needed and not already present
    include_added = False
    if needs_include and not already_has_include(lines):
        insert_idx = find_include_insertion_point(lines)
        include_line = f'#include "{HEADER}"\n'
        lines.insert(insert_idx, include_line)
        include_added = True

    # Write if changed
    if defines_removed > 0 or literals_replaced > 0:
        if not DRY_RUN:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)

    return defines_removed, literals_replaced, include_added


def main():
    total_defines = 0
    total_literals = 0
    total_includes = 0
    files_modified = 0

    for search_dir in [SRC_DIR, INCLUDE_DIR]:
        for root, dirs, files in os.walk(search_dir):
            # Skip test directories
            if '/test/' in root or root.endswith('/test'):
                continue
            for fname in sorted(files):
                if not fname.endswith(('.c', '.h', '.cu')):
                    continue
                filepath = os.path.join(root, fname)
                d, l, inc = process_file(filepath)
                if d > 0 or l > 0:
                    files_modified += 1
                    total_defines += d
                    total_literals += l
                    if inc:
                        total_includes += 1
                    if DRY_RUN:
                        print(f"  {filepath}: {d} defines removed, {l} literals replaced")

    mode = "[DRY RUN] " if DRY_RUN else ""
    print(f"\n{mode}Math constants replacement complete:")
    print(f"  Files modified:    {files_modified}")
    print(f"  Defines removed:   {total_defines}")
    print(f"  Literals replaced: {total_literals}")
    print(f"  Includes added:    {total_includes}")
    print(f"  Total changes:     {total_defines + total_literals}")


if __name__ == "__main__":
    main()
