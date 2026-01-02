#!/usr/bin/env python3
"""
GPU Common Migration Script

Scans all .cu files in src/gpu/ and migrates duplicate patterns to use
the standardized nimcp_gpu_common.h library.

Replacements:
  - device_sigmoid, gpu_sigmoid -> nimcp_device_sigmoid
  - device_tanh, gpu_tanh -> nimcp_device_tanh
  - device_relu -> nimcp_device_relu
  - device_gelu, gelu_device -> nimcp_device_gelu
  - device_silu -> nimcp_device_silu
  - device_softplus -> nimcp_device_softplus
  - device_clamp -> nimcp_device_clamp
  - warp_reduce_sum -> nimcp_warp_reduce_sum
  - warp_reduce_max -> nimcp_warp_reduce_max
  - CUDA_CHECK, CHECK_CUDA_ERROR -> NIMCP_CUDA_CHECK
  - BLOCK_SIZE -> NIMCP_CUDA_BLOCK_SIZE
  - GRID_SIZE -> NIMCP_CUDA_GRID_SIZE

Features:
  - --dry-run: Preview changes without modifying files
  - --verbose: Detailed output
  - --backup: Create .bak files before modifying
  - Comments out old function definitions (doesn't delete)
  - Adds #include "gpu/common/nimcp_gpu_common.h" if not present
  - Generates migration report

Author: NIMCP Development Team
Date: 2025
"""

import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

# Root directory for GPU source files
GPU_SRC_ROOT = "/home/bbrelin/nimcp/src/gpu"

# The common header to include
COMMON_HEADER = '#include "gpu/common/nimcp_gpu_common.h"'

# Function name replacements: old_name -> new_name
# Note: We use word boundaries to avoid partial matches
FUNCTION_REPLACEMENTS = {
    # Activation functions
    "device_sigmoid": "nimcp_device_sigmoid",
    "gpu_sigmoid": "nimcp_device_sigmoid",
    "device_tanh": "nimcp_device_tanh",
    "gpu_tanh": "nimcp_device_tanh",
    "device_relu": "nimcp_device_relu",
    "device_gelu": "nimcp_device_gelu",
    "gelu_device": "nimcp_device_gelu",
    "device_silu": "nimcp_device_silu",
    "device_softplus": "nimcp_device_softplus",
    "device_clamp": "nimcp_device_clamp",
    # Warp primitives
    "warp_reduce_sum": "nimcp_warp_reduce_sum",
    "warp_reduce_max": "nimcp_warp_reduce_max",
}

# Macro replacements: old_name -> new_name
MACRO_REPLACEMENTS = {
    "CUDA_CHECK": "NIMCP_CUDA_CHECK",
    "CHECK_CUDA_ERROR": "NIMCP_CUDA_CHECK",
    "BLOCK_SIZE": "NIMCP_CUDA_BLOCK_SIZE",
    "GRID_SIZE": "NIMCP_CUDA_GRID_SIZE",
}

# Patterns for function definitions that should be commented out
# These match __device__ inline? function definitions
FUNCTION_DEF_PATTERNS = [
    # __device__ inline float device_sigmoid(float x) { ... }
    r'(__device__\s+(?:inline\s+)?(?:float|int|double|void)\s+(?:device_sigmoid|gpu_sigmoid|device_tanh|gpu_tanh|device_relu|device_gelu|gelu_device|device_silu|device_softplus|device_clamp|warp_reduce_sum|warp_reduce_max)\s*\([^)]*\))',
    # __device__ float device_sigmoid(float x) { ... }
    r'(__device__\s+(?:float|int|double|void)\s+(?:device_sigmoid|gpu_sigmoid|device_tanh|gpu_tanh|device_relu|device_gelu|gelu_device|device_silu|device_softplus|device_clamp|warp_reduce_sum|warp_reduce_max)\s*\([^)]*\))',
]

# Patterns for macro definitions that should be commented out
MACRO_DEF_PATTERNS = [
    # #define CUDA_CHECK(call) ...
    r'^(\s*#define\s+(?:CUDA_CHECK|CHECK_CUDA_ERROR|BLOCK_SIZE|GRID_SIZE)\b)',
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FileChange:
    """Represents changes made to a single file."""
    filepath: str
    include_added: bool = False
    functions_commented: List[str] = field(default_factory=list)
    macros_commented: List[str] = field(default_factory=list)
    function_calls_replaced: Dict[str, int] = field(default_factory=dict)
    macro_uses_replaced: Dict[str, int] = field(default_factory=dict)
    lines_modified: int = 0
    backup_created: bool = False
    error: Optional[str] = None


@dataclass
class MigrationReport:
    """Overall migration report."""
    start_time: datetime
    end_time: Optional[datetime] = None
    files_scanned: int = 0
    files_modified: int = 0
    files_skipped: int = 0
    files_with_errors: int = 0
    total_includes_added: int = 0
    total_functions_commented: int = 0
    total_macros_commented: int = 0
    total_function_calls_replaced: int = 0
    total_macro_uses_replaced: int = 0
    file_changes: List[FileChange] = field(default_factory=list)
    dry_run: bool = False


# =============================================================================
# Migration Functions
# =============================================================================

def find_cu_files(root_dir: str) -> List[str]:
    """Find all .cu files recursively under root_dir."""
    cu_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.cu'):
                cu_files.append(os.path.join(dirpath, filename))
    return sorted(cu_files)


def has_common_header(content: str) -> bool:
    """Check if the file already includes nimcp_gpu_common.h."""
    patterns = [
        r'#include\s+"gpu/common/nimcp_gpu_common\.h"',
        r'#include\s+<gpu/common/nimcp_gpu_common\.h>',
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False


def add_common_header(content: str) -> Tuple[str, bool]:
    """
    Add #include "gpu/common/nimcp_gpu_common.h" to the file.

    Strategy:
    1. Find the last #include statement in the initial include block
    2. Add the common header after it

    Returns (new_content, was_added)
    """
    if has_common_header(content):
        return content, False

    lines = content.split('\n')

    # Find the best position to insert the include
    # Look for the last #include in the initial block (before any code)
    last_include_idx = -1
    in_multiline_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track multiline comments
        if '/*' in stripped and '*/' not in stripped:
            in_multiline_comment = True
            continue
        if '*/' in stripped:
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            continue

        # Skip single-line comments and empty lines
        if stripped.startswith('//') or stripped.startswith('/*') or not stripped:
            continue

        # Check for include statements
        if stripped.startswith('#include'):
            last_include_idx = i
        # Check for preprocessor directives (ifdef, ifndef, define, etc.)
        elif stripped.startswith('#'):
            # Allow preprocessor directives before includes
            continue
        # If we hit actual code, stop
        elif not stripped.startswith('//') and not stripped.startswith('*'):
            # Only break if we've seen at least one include
            if last_include_idx >= 0:
                break

    if last_include_idx >= 0:
        # Insert after the last include
        lines.insert(last_include_idx + 1, COMMON_HEADER)
        return '\n'.join(lines), True
    else:
        # No includes found, add at the beginning after any initial comments
        # Find the first non-comment, non-empty line
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                insert_idx = i
                break

        lines.insert(insert_idx, COMMON_HEADER)
        lines.insert(insert_idx + 1, '')  # Add blank line after
        return '\n'.join(lines), True


def find_function_definitions(content: str) -> List[Tuple[int, int, str]]:
    """
    Find function definitions that should be commented out.

    Returns list of (start_line, end_line, function_name) tuples.
    """
    definitions = []
    lines = content.split('\n')

    # Build a combined pattern for function names
    func_names = '|'.join([
        'device_sigmoid', 'gpu_sigmoid', 'device_tanh', 'gpu_tanh',
        'device_relu', 'device_gelu', 'gelu_device', 'device_silu',
        'device_softplus', 'device_clamp', 'warp_reduce_sum', 'warp_reduce_max'
    ])

    # Pattern to match __device__ function definition start
    func_start_pattern = re.compile(
        rf'__device__\s+(?:inline\s+)?(?:float|int|double|void|__forceinline__\s+float)\s+({func_names})\s*\('
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        match = func_start_pattern.search(line)

        if match:
            func_name = match.group(1)
            start_line = i

            # Find the end of the function
            # Look for matching braces
            brace_count = 0
            found_open_brace = False
            end_line = start_line

            for j in range(start_line, len(lines)):
                for char in lines[j]:
                    if char == '{':
                        brace_count += 1
                        found_open_brace = True
                    elif char == '}':
                        brace_count -= 1

                if found_open_brace and brace_count == 0:
                    end_line = j
                    break

            definitions.append((start_line, end_line, func_name))
            i = end_line + 1
        else:
            i += 1

    return definitions


def find_macro_definitions(content: str) -> List[Tuple[int, int, str]]:
    """
    Find macro definitions that should be commented out.

    Returns list of (start_line, end_line, macro_name) tuples.
    """
    definitions = []
    lines = content.split('\n')

    macro_names = ['CUDA_CHECK', 'CHECK_CUDA_ERROR', 'BLOCK_SIZE', 'GRID_SIZE']

    for i, line in enumerate(lines):
        stripped = line.strip()

        for macro_name in macro_names:
            # Match #define MACRO_NAME (potentially with continuation)
            if re.match(rf'#\s*define\s+{macro_name}\b', stripped):
                start_line = i
                end_line = i

                # Handle multi-line macros (ending with \)
                while end_line < len(lines) and lines[end_line].rstrip().endswith('\\'):
                    end_line += 1

                definitions.append((start_line, end_line, macro_name))
                break

    return definitions


def comment_out_lines(lines: List[str], start: int, end: int, label: str) -> List[str]:
    """Comment out lines from start to end (inclusive)."""
    result = lines.copy()

    # Add comment header
    result[start] = f"/* MIGRATED TO nimcp_gpu_common.h - {label}\n" + result[start]
    result[end] = result[end] + "\n*/"

    return result


def replace_function_calls(content: str, replacements: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """
    Replace function calls with new names.

    Uses word boundaries to avoid partial matches.
    Returns (new_content, replacement_counts)
    """
    counts = {}

    for old_name, new_name in replacements.items():
        # Use word boundaries to match only complete function names
        # But don't match in comments or strings
        pattern = rf'\b{re.escape(old_name)}\b'

        # Count occurrences
        matches = re.findall(pattern, content)
        if matches:
            # Exclude matches that are part of the new name (already migrated)
            # or in definition contexts
            actual_count = 0
            for m in re.finditer(pattern, content):
                # Check if this is inside a comment or string
                start = m.start()
                # Simple heuristic: check if preceded by nimcp_
                prefix_start = max(0, start - 10)
                prefix = content[prefix_start:start]
                if 'nimcp_' not in prefix and new_name not in content[start:start+len(new_name)+10]:
                    actual_count += 1

            if actual_count > 0:
                content = re.sub(pattern, new_name, content)
                counts[old_name] = actual_count

    return content, counts


def replace_macro_uses(content: str, replacements: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """
    Replace macro uses with new names.

    Be careful not to replace the macro in its own definition or
    in the common header include.

    Returns (new_content, replacement_counts)
    """
    counts = {}

    for old_name, new_name in replacements.items():
        # For macros like BLOCK_SIZE and GRID_SIZE, we need to be more careful
        # Only replace when used in kernel launch configs or similar contexts

        if old_name in ['BLOCK_SIZE', 'GRID_SIZE']:
            # Match BLOCK_SIZE or GRID_SIZE but not in #define contexts
            # and not if already NIMCP_ prefixed
            pattern = rf'(?<!NIMCP_)(?<!#define\s)(?<!\w){re.escape(old_name)}\b'
        else:
            # For CUDA_CHECK, match anywhere except in #define and already prefixed
            pattern = rf'(?<!NIMCP_)(?<!#define\s)\b{re.escape(old_name)}\b'

        matches = list(re.finditer(pattern, content))

        if matches:
            actual_count = 0
            new_content = content
            offset = 0

            for m in matches:
                start = m.start() + offset
                end = m.end() + offset

                # Double-check this isn't in a #define line
                line_start = new_content.rfind('\n', 0, start) + 1
                line = new_content[line_start:start]

                if '#define' not in line and 'NIMCP_' not in new_content[max(0, start-6):start]:
                    new_content = new_content[:start] + new_name + new_content[end:]
                    offset += len(new_name) - len(old_name)
                    actual_count += 1

            if actual_count > 0:
                content = new_content
                counts[old_name] = actual_count

    return content, counts


def process_file(filepath: str, dry_run: bool = False, backup: bool = False,
                 verbose: bool = False) -> FileChange:
    """
    Process a single .cu file.

    Returns FileChange describing what was done.
    """
    change = FileChange(filepath=filepath)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()
    except Exception as e:
        change.error = f"Failed to read file: {e}"
        return change

    content = original_content
    lines = content.split('\n')

    # 1. Add common header if needed
    content, include_added = add_common_header(content)
    change.include_added = include_added

    # 2. Find and comment out function definitions
    func_defs = find_function_definitions(content)
    if func_defs:
        lines = content.split('\n')
        # Process in reverse order to maintain line numbers
        for start, end, name in reversed(func_defs):
            lines = comment_out_lines(lines, start, end, name)
            change.functions_commented.append(name)
        content = '\n'.join(lines)

    # 3. Find and comment out macro definitions
    macro_defs = find_macro_definitions(content)
    if macro_defs:
        lines = content.split('\n')
        # Process in reverse order to maintain line numbers
        for start, end, name in reversed(macro_defs):
            lines = comment_out_lines(lines, start, end, name)
            change.macros_commented.append(name)
        content = '\n'.join(lines)

    # 4. Replace function calls
    content, func_counts = replace_function_calls(content, FUNCTION_REPLACEMENTS)
    change.function_calls_replaced = func_counts

    # 5. Replace macro uses
    content, macro_counts = replace_macro_uses(content, MACRO_REPLACEMENTS)
    change.macro_uses_replaced = macro_counts

    # Calculate lines modified
    original_lines = original_content.split('\n')
    new_lines = content.split('\n')
    change.lines_modified = sum(1 for o, n in zip(original_lines, new_lines) if o != n)
    change.lines_modified += abs(len(new_lines) - len(original_lines))

    # Check if anything changed
    if content != original_content:
        if not dry_run:
            # Create backup if requested
            if backup:
                backup_path = filepath + '.bak'
                try:
                    shutil.copy2(filepath, backup_path)
                    change.backup_created = True
                except Exception as e:
                    change.error = f"Failed to create backup: {e}"
                    return change

            # Write the modified content
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                change.error = f"Failed to write file: {e}"
                return change

    return change


def generate_report(report: MigrationReport, verbose: bool = False) -> str:
    """Generate a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("GPU COMMON MIGRATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    if report.dry_run:
        lines.append("*** DRY RUN MODE - No files were modified ***")
        lines.append("")

    lines.append(f"Start Time: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if report.end_time:
        lines.append(f"End Time: {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = (report.end_time - report.start_time).total_seconds()
        lines.append(f"Duration: {duration:.2f} seconds")
    lines.append("")

    lines.append("-" * 40)
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Files scanned:              {report.files_scanned}")
    lines.append(f"Files modified:             {report.files_modified}")
    lines.append(f"Files skipped (no changes): {report.files_skipped}")
    lines.append(f"Files with errors:          {report.files_with_errors}")
    lines.append("")
    lines.append(f"Total includes added:           {report.total_includes_added}")
    lines.append(f"Total functions commented out:  {report.total_functions_commented}")
    lines.append(f"Total macros commented out:     {report.total_macros_commented}")
    lines.append(f"Total function calls replaced:  {report.total_function_calls_replaced}")
    lines.append(f"Total macro uses replaced:      {report.total_macro_uses_replaced}")
    lines.append("")

    # List files with changes
    modified_files = [c for c in report.file_changes if c.lines_modified > 0 or c.error]

    if modified_files:
        lines.append("-" * 40)
        lines.append("FILES WITH CHANGES")
        lines.append("-" * 40)

        for change in modified_files:
            rel_path = os.path.relpath(change.filepath, GPU_SRC_ROOT)
            lines.append("")
            lines.append(f"  {rel_path}")

            if change.error:
                lines.append(f"    ERROR: {change.error}")
                continue

            if change.include_added:
                lines.append("    + Added #include for nimcp_gpu_common.h")

            if change.functions_commented:
                lines.append(f"    - Commented out functions: {', '.join(change.functions_commented)}")

            if change.macros_commented:
                lines.append(f"    - Commented out macros: {', '.join(change.macros_commented)}")

            if change.function_calls_replaced:
                for old, count in change.function_calls_replaced.items():
                    new = FUNCTION_REPLACEMENTS.get(old, old)
                    lines.append(f"    > Replaced {old} -> {new} ({count} occurrences)")

            if change.macro_uses_replaced:
                for old, count in change.macro_uses_replaced.items():
                    new = MACRO_REPLACEMENTS.get(old, old)
                    lines.append(f"    > Replaced {old} -> {new} ({count} occurrences)")

            if change.backup_created:
                lines.append("    * Backup created (.bak)")

    # Verbose: show all files
    if verbose:
        skipped_files = [c for c in report.file_changes if c.lines_modified == 0 and not c.error]
        if skipped_files:
            lines.append("")
            lines.append("-" * 40)
            lines.append("FILES SKIPPED (no changes needed)")
            lines.append("-" * 40)
            for change in skipped_files:
                rel_path = os.path.relpath(change.filepath, GPU_SRC_ROOT)
                lines.append(f"  {rel_path}")

    lines.append("")
    lines.append("=" * 80)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate GPU CUDA files to use nimcp_gpu_common.h",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run              Preview changes without modifying files
  %(prog)s --verbose              Show detailed output
  %(prog)s --backup               Create .bak files before modifying
  %(prog)s --dry-run --verbose    Preview with details
        """
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without modifying files'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help='Create .bak backup files before modifying'
    )

    parser.add_argument(
        '--path', '-p',
        default=GPU_SRC_ROOT,
        help=f'Root path to scan (default: {GPU_SRC_ROOT})'
    )

    args = parser.parse_args()

    # Initialize report
    report = MigrationReport(
        start_time=datetime.now(),
        dry_run=args.dry_run
    )

    # Find all .cu files
    print(f"Scanning for .cu files in {args.path}...")
    cu_files = find_cu_files(args.path)
    report.files_scanned = len(cu_files)
    print(f"Found {len(cu_files)} .cu files")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")

    # Process each file
    for filepath in cu_files:
        if args.verbose:
            rel_path = os.path.relpath(filepath, args.path)
            print(f"Processing: {rel_path}")

        change = process_file(
            filepath,
            dry_run=args.dry_run,
            backup=args.backup,
            verbose=args.verbose
        )

        report.file_changes.append(change)

        # Update totals
        if change.error:
            report.files_with_errors += 1
        elif change.lines_modified > 0:
            report.files_modified += 1
            report.total_includes_added += 1 if change.include_added else 0
            report.total_functions_commented += len(change.functions_commented)
            report.total_macros_commented += len(change.macros_commented)
            report.total_function_calls_replaced += sum(change.function_calls_replaced.values())
            report.total_macro_uses_replaced += sum(change.macro_uses_replaced.values())
        else:
            report.files_skipped += 1

    report.end_time = datetime.now()

    # Generate and print report
    print("\n")
    print(generate_report(report, verbose=args.verbose))

    # Exit with appropriate code
    if report.files_with_errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
