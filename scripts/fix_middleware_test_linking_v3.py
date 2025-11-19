#!/usr/bin/env python3
"""
Fix middleware test CMakeLists.txt files to link properly.

Middleware tests need to link against BOTH libraries:
- nimcp_middleware (for middleware functions)
- nimcp (for utility functions that middleware uses)
- GTest::GTest and GTest::Main (testing framework)
"""

import os
import re
from pathlib import Path

def fix_cmake_file(filepath):
    """Fix a single CMakeLists.txt file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace target_link_libraries blocks
    pattern = r'(target_link_libraries\([^\)]+?\s+PRIVATE\s*\n)(.*?)(\n\s*\))'

    def replace_libs(match):
        prefix = match.group(1)
        libs = match.group(2)
        suffix = match.group(3)

        # Standard libraries for middleware tests
        # Order matters: nimcp_middleware first, then nimcp (for transitive dependencies)
        new_libs = [
            '        GTest::GTest',
            '        GTest::Main',
            '        nimcp_middleware',
            '        nimcp'
        ]

        return f"{prefix}{chr(10).join(new_libs)}{suffix}"

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
