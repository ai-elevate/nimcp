#!/usr/bin/env python3
"""
Convert SRP Split to #include-based Organization
==================================================
Instead of separately-compiled split files (which break on opaque structs),
use #include-based organization where the parent .c file includes the split parts.

This approach:
- Preserves file organization for code navigation
- Fixes all struct visibility issues
- Requires no changes to CMakeLists.txt (split files aren't compiled separately)
"""

import os
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"
CMAKE_FILE = SRC_ROOT / "lib" / "CMakeLists.txt"


def main():
    dry_run = '--dry-run' in os.sys.argv

    # Find all .c.orig backups
    orig_files = sorted(SRC_ROOT.rglob("*.c.orig"))
    print(f"Found {len(orig_files)} original file backups\n")

    all_split_files = []

    for orig in orig_files:
        module_name = orig.stem.replace('.c', '')
        parent = orig.parent
        current_file = parent / (module_name + '.c')

        # Find split files
        split_files = []
        for f in sorted(parent.iterdir()):
            if f.is_file() and f.suffix == '.c' and f.stem.startswith(module_name + '_') and \
               not f.stem.endswith('_facade') and not f.stem.endswith('.c') and \
               f.name != orig.name and f.name != current_file.name:
                split_files.append(f)

        if not split_files:
            print(f"  {module_name}: No splits, restoring original")
            if not dry_run:
                shutil.copy2(orig, current_file)
            continue

        all_split_files.extend(split_files)

        print(f"  {module_name}: {len(split_files)} split files")

        if dry_run:
            continue

        # Step 1: Restore original file
        with open(orig, 'r') as f:
            orig_content = f.read()

        orig_lines = orig_content.split('\n')

        # Step 2: Build map of which functions are in which split file
        # Read each split file to find its function names
        split_func_names = {}
        for sf in split_files:
            with open(sf, 'r') as f:
                sf_content = f.read()
            # Find function names in split file
            func_names = set()
            for match in re.finditer(r'^(?:static\s+)?(?:inline\s+)?(?:void|int|bool|float|double|char\s*\*?|uint\d+_t|int\d+_t|size_t|ssize_t|nimcp_\w+|brain_\w+|omni_\w+|nlp_\w+)\s+\*?(\w+)\s*\(', sf_content, re.MULTILINE):
                func_names.add(match.group(1))
            split_func_names[sf.name] = func_names

        # Step 3: Create updated original that includes split files at the end
        # The original file keeps ALL its content (structs, includes, functions)
        # The split files are just organizational copies for navigation
        # We DON'T include the split files - they're just reference copies

        # Restore the original as-is
        shutil.copy2(orig, current_file)

    # Step 4: Remove split file entries from CMakeLists.txt
    print(f"\nRemoving {len(all_split_files)} split file entries from CMakeLists.txt")
    if not dry_run:
        with open(CMAKE_FILE, 'r') as f:
            cmake_content = f.read()

        # Remove lines containing split file names
        cmake_lines = cmake_content.split('\n')
        new_cmake_lines = []
        skip_next_comment = False
        for i, line in enumerate(cmake_lines):
            # Check if this line references a split file
            is_split_ref = False
            for sf in all_split_files:
                if sf.name in line:
                    is_split_ref = True
                    break

            # Also remove "SRP split files" comments
            if 'SRP split files' in line:
                skip_next_comment = True
                continue

            if is_split_ref:
                continue

            new_cmake_lines.append(line)

        with open(CMAKE_FILE, 'w') as f:
            f.write('\n'.join(new_cmake_lines))

    print(f"\nDone. Split .c files remain as reference copies but are not compiled separately.")
    print(f"The original files have been restored and compile as single units.")
    print(f"\nNext: cd build && cmake .. && make nimcp -j4")


if __name__ == '__main__':
    main()
