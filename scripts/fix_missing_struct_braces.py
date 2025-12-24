#!/usr/bin/env python3
"""
Fix structs that are missing closing braces after deprecated field removal.

The pattern: struct starts with `struct X {`, but ends with just a field
before a comment block or function declaration, missing the `};`
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def fix_missing_braces(filepath):
    """Add missing closing braces to structs."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern: a struct field line (with semicolon) directly followed by
    # a comment block starting a new section (/* ===...) without a closing brace
    pattern = r'(\s+\S+[^;]*;\s*(?:/\*\*?[^*]*\*/)?)\n(/\* =+)'

    def add_closing_brace(match):
        field_line = match.group(1)
        comment_start = match.group(2)
        return f"{field_line}\n}};\n\n{comment_start}"

    new_content, count = re.subn(pattern, add_closing_brace, content)

    if count > 0 and new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True

    return False

def main():
    print("=" * 60)
    print("Fixing missing struct closing braces")
    print("=" * 60)

    fixed_count = 0

    # Process headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if fix_missing_braces(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} headers")

if __name__ == '__main__':
    main()
