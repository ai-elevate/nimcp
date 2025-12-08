#!/usr/bin/env python3
"""
Security Validation Integration Script

This script adds security validation to NIMCP source files that handle
external input but don't have security registration yet.

WHAT: Automates security header inclusion and validation calls
WHY: Increase security coverage from 8.5% to 30%+
HOW: Parse C files, identify entry points, inject validation code
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Files that already have security (skip these)
ALREADY_SECURED = {
    'nimcp_stream.c',
    'nimcp_dataio.c',
    'nimcp_distributed_cognition.c',
    'nimcp_distributed_cognition_refactored.c',
    'nimcp_protocol.c',
    'nimcp_brain_learning.c',
    'nimcp_security.c',
    'nimcp_blood_brain_barrier.c',
    'nimcp_bbb_input_gate.c',
}

# High priority modules that handle external input
HIGH_PRIORITY_PATTERNS = [
    'serialization',
    'encryption',
    'network_serialization',
    'p2pnode',
    'events',
    'replication',
    'event_queue',
    'event_bus',
    'event_subscriber',
    'routing',
    'signal_wrapper',
]

class SecurityInjector:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.lines = []
        self.modified = False

    def read_file(self) -> bool:
        """Read the source file"""
        try:
            with open(self.file_path, 'r') as f:
                self.lines = f.readlines()
            return True
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            return False

    def has_security_include(self) -> bool:
        """Check if file already includes security headers"""
        content = ''.join(self.lines)
        return ('security/nimcp_security.h' in content or
                'security/nimcp_blood_brain_barrier.h' in content or
                'security/nimcp_security_integration.h' in content)

    def find_include_section_end(self) -> int:
        """Find the line after the last #include statement"""
        last_include_line = -1
        for i, line in enumerate(self.lines):
            if line.strip().startswith('#include'):
                last_include_line = i
        return last_include_line + 1 if last_include_line >= 0 else 0

    def add_security_include(self) -> bool:
        """Add security header include if not present"""
        if self.has_security_include():
            return False

        # Find where to insert
        insert_pos = self.find_include_section_end()
        if insert_pos == 0:
            print(f"Warning: No includes found in {self.file_name}")
            return False

        # Add security include
        self.lines.insert(insert_pos, '#include "security/nimcp_blood_brain_barrier.h"\n')
        self.modified = True
        return True

    def find_external_input_functions(self) -> List[Tuple[int, str]]:
        """
        Find functions that likely handle external input based on:
        - Function name patterns (read, recv, load, parse, deserialize, etc.)
        - Parameter types (char*, uint8_t*, void*, size_t)
        """
        input_funcs = []

        # Patterns that suggest external input handling
        input_keywords = [
            r'\bread\w*\(',
            r'\brecv\w*\(',
            r'\bload\w*\(',
            r'\bparse\w*\(',
            r'\bdeserialize\w*\(',
            r'\bprocess\w*\(',
            r'\bhandle\w*\(',
            r'\bset_buffer\(',
            r'\bnext_batch\(',
            r'\bconnect\w*\(',
        ]

        for i, line in enumerate(self.lines):
            # Look for function definitions with suspicious patterns
            if any(re.search(pattern, line) for pattern in input_keywords):
                # Check if it's a function definition (has { or is followed by {)
                if '{' in line or (i + 1 < len(self.lines) and '{' in self.lines[i + 1]):
                    func_match = re.search(r'(\w+)\s*\([^)]*\)', line)
                    if func_match:
                        func_name = func_match.group(1)
                        input_funcs.append((i, func_name))

        return input_funcs

    def add_input_validation(self, func_line: int, func_name: str) -> bool:
        """Add BBB input validation at the start of a function"""
        # Find the opening brace
        brace_line = func_line
        for i in range(func_line, min(func_line + 5, len(self.lines))):
            if '{' in self.lines[i]:
                brace_line = i
                break

        # Check if validation already exists nearby
        check_range = range(brace_line + 1, min(brace_line + 10, len(self.lines)))
        for i in check_range:
            if 'bbb_validate' in self.lines[i] or 'nimcp_security_validate' in self.lines[i]:
                return False  # Already has validation

        # Insert validation code after opening brace
        indent = '    '  # Standard 4-space indent
        validation_code = f'''
{indent}// BBB: Validate external input
{indent}// WHAT: Check input for security threats before processing
{indent}// WHY: Prevent injection attacks and buffer overflows
{indent}// TODO: Customize validation for specific input parameters
{indent}/*
{indent}bbb_validation_result_t val_result = {{0}};
{indent}if (!bbb_validate_input(system, data, size, &val_result)) {{
{indent}    NIMCP_LOG_ERROR("Input validation failed in {func_name}: %s", val_result.reason);
{indent}    return NIMCP_ERROR_INVALID_INPUT;
{indent}}}
{indent}*/
'''
        self.lines.insert(brace_line + 1, validation_code)
        self.modified = True
        return True

    def inject_security(self) -> bool:
        """Main injection logic"""
        if not self.read_file():
            return False

        # Step 1: Add security include
        include_added = self.add_security_include()

        # Step 2: Find and annotate input functions
        input_funcs = self.find_external_input_functions()
        validations_added = 0

        for func_line, func_name in input_funcs:
            if self.add_input_validation(func_line, func_name):
                validations_added += 1

        if self.modified:
            print(f"✓ {self.file_name}:")
            if include_added:
                print(f"  - Added security header include")
            if validations_added > 0:
                print(f"  - Added {validations_added} validation annotations")

        return self.modified

    def write_file(self) -> bool:
        """Write modified file back to disk"""
        if not self.modified:
            return False

        try:
            with open(self.file_path, 'w') as f:
                f.writelines(self.lines)
            return True
        except Exception as e:
            print(f"Error writing {self.file_path}: {e}")
            return False


def should_process_file(file_path: str) -> bool:
    """Determine if file should be processed"""
    file_name = os.path.basename(file_path)

    # Skip if already secured
    if file_name in ALREADY_SECURED:
        return False

    # Skip backup files
    if file_name.endswith('.backup') or file_name.endswith('~'):
        return False

    # Process if high priority
    for pattern in HIGH_PRIORITY_PATTERNS:
        if pattern in file_name:
            return True

    return False


def find_target_files(base_dir: str) -> List[str]:
    """Find all C files that need security injection"""
    targets = []

    priority_dirs = [
        'src/io/serialization',
        'src/io/stream',
        'src/networking/p2p',
        'src/networking/events',
        'src/networking/replication',
        'src/networking/protocol',
        'src/middleware/events',
        'src/middleware/routing',
        'src/middleware/buffering',
    ]

    for dir_path in priority_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            continue

        for file_name in os.listdir(full_path):
            if file_name.endswith('.c'):
                file_path = os.path.join(full_path, file_name)
                if should_process_file(file_path):
                    targets.append(file_path)

    return sorted(targets)


def main():
    # Get NIMCP root directory
    if len(sys.argv) > 1:
        nimcp_root = sys.argv[1]
    else:
        # Assume script is in scripts/ directory
        nimcp_root = Path(__file__).parent.parent

    print("="*70)
    print("NIMCP Security Validation Injection")
    print("="*70)
    print(f"Base directory: {nimcp_root}\n")

    # Find target files
    targets = find_target_files(nimcp_root)
    print(f"Found {len(targets)} files to process\n")

    if len(targets) == 0:
        print("No files found. Check that base directory is correct.")
        return 1

    # Process each file
    modified_count = 0
    for file_path in targets:
        injector = SecurityInjector(file_path)
        if injector.inject_security():
            injector.write_file()
            modified_count += 1

    # Summary
    print("\n" + "="*70)
    print(f"Summary: Modified {modified_count}/{len(targets)} files")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the TODO comments in modified files")
    print("2. Customize validation for specific input parameters")
    print("3. Add proper error handling and logging")
    print("4. Test the security validations")
    print("5. Remove TODO comments once validation is complete")

    return 0


if __name__ == '__main__':
    sys.exit(main())
