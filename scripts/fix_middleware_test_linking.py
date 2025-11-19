#!/usr/bin/env python3
"""
Fix middleware test CMakeLists.txt files to link against nimcp library.

All middleware test executables need to link against the main nimcp library
(which contains all util functions) rather than just nimcp_middleware.
"""

import os
import re
from pathlib import Path

def fix_cmake_file(filepath):
    """Fix a single CMakeLists.txt file to link against nimcp."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace target_link_libraries pattern:
    # - Keep GTest::GTest and GTest::Main
    # - Add nimcp library
    # - Keep nimcp_middleware if present

    # Pattern: find target_link_libraries blocks
    pattern = r'(target_link_libraries\([^\)]+?\s+PRIVATE\s+)(.*?)(\))'

    def replace_libs(match):
        prefix = match.group(1)
        libs = match.group(2)
        suffix = match.group(3)

        # Parse existing libraries
        lib_list = [lib.strip() for lib in libs.split('\n') if lib.strip()]

        # Ensure we have the correct libraries
        new_libs = []
        has_gtest = False
        has_gtest_main = False
        has_nimcp = False
        has_middleware = False

        for lib in lib_list:
            if 'GTest::GTest' in lib and 'Main' not in lib:
                has_gtest = True
                new_libs.append('GTest::GTest')
            elif 'GTest::Main' in lib:
                has_gtest_main = True
                new_libs.append('GTest::Main')
            elif lib == 'nimcp':
                has_nimcp = True
            elif lib == 'nimcp_middleware':
                has_middleware = True
            elif lib.startswith('nimcp_utils'):
                # Remove utils libraries - they're in nimcp
                continue
            elif lib and not lib.startswith('#'):
                new_libs.append(lib)

        # Add required libraries
        if not has_gtest:
            new_libs.insert(0, 'GTest::GTest')
        if not has_gtest_main:
            new_libs.insert(1, 'GTest::Main')
        if not has_nimcp:
            new_libs.append('nimcp')
        if not has_middleware:
            new_libs.append('nimcp_middleware')

        # Format output
        libs_str = '\n        '.join(new_libs)
        return f"{prefix}\n        {libs_str}\n    {suffix}"

    new_content = re.sub(pattern, replace_libs, content, flags=re.DOTALL)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"Fixed: {filepath}")

def main():
    test_dir = Path("/home/bbrelin/nimcp/test")

    # Find all middleware test CMakeLists.txt files
    patterns = [
        "unit/middleware/**/CMakeLists.txt",
        "integration/middleware/**/CMakeLists.txt",
        "regression/middleware/**/CMakeLists.txt"
    ]

    for pattern in patterns:
        for filepath in test_dir.glob(pattern):
            if filepath.is_file():
                fix_cmake_file(filepath)

if __name__ == "__main__":
    main()
