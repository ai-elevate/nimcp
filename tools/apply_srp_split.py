#!/usr/bin/env python3
"""
Apply SRP Split: Replace originals with facades and update CMakeLists.txt
=========================================================================
This script:
1. Backs up original .c files
2. Replaces them with the facade versions
3. Updates CMakeLists.txt to include the new split files

Usage:
    python3 tools/apply_srp_split.py [--dry-run]
"""

import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"
CMAKE_FILE = SRC_ROOT / "lib" / "CMakeLists.txt"

# Files that were already split in a previous session (don't touch these)
ALREADY_SPLIT = {
    "src/utils/thread/nimcp_thread.c",  # Already split into 6 files
    "src/swarm/nimcp_swarm_consciousness_enhanced.c",  # Already split into 4 files
}


def find_facade_pairs():
    """Find all (original.c, original_facade.c) pairs."""
    pairs = []
    for facade in sorted(SRC_ROOT.rglob("*_facade.c")):
        # Derive original name from facade name
        original_name = facade.name.replace("_facade.c", ".c")
        original_path = facade.parent / original_name

        rel_path = str(original_path.relative_to(PROJECT_ROOT))
        if rel_path in ALREADY_SPLIT:
            print(f"  SKIP (already split): {rel_path}")
            continue

        if original_path.exists():
            pairs.append((original_path, facade))
        else:
            print(f"  WARNING: No original for facade {facade}")
    return pairs


def find_split_files(original_path):
    """Find all split .c files that correspond to an original."""
    module_name = original_path.stem  # e.g., nimcp_brain_immune
    parent = original_path.parent
    splits = []
    for f in sorted(parent.iterdir()):
        if f.is_file() and f.suffix == '.c':
            # Match: module_name_SUFFIX.c but NOT module_name_facade.c
            if f.stem.startswith(module_name + '_') and not f.stem.endswith('_facade'):
                # Exclude the original itself
                if f != original_path:
                    splits.append(f)
    return splits


def update_cmake(pairs, dry_run=False):
    """Update CMakeLists.txt to add split files."""
    with open(CMAKE_FILE, 'r') as f:
        cmake_content = f.read()

    cmake_lines = cmake_content.split('\n')
    new_lines = []

    for pair_original, pair_facade in pairs:
        splits = find_split_files(pair_original)
        if not splits:
            continue

        # Find the line in CMakeLists.txt that references the original
        original_rel = os.path.relpath(pair_original, SRC_ROOT / "lib")
        # CMakeLists uses ${CMAKE_CURRENT_SOURCE_DIR}/..
        cmake_ref = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{original_rel}"

        found = False
        for i, line in enumerate(cmake_lines):
            if cmake_ref in line or str(pair_original.name) in line:
                # Found the reference - add split file references after it
                # First, add a comment
                indent = '    '
                comment = f"{indent}# SRP split files for {pair_original.name}:"
                split_entries = []
                for split in splits:
                    split_rel = os.path.relpath(split, SRC_ROOT / "lib")
                    split_cmake = f"{indent}${{CMAKE_CURRENT_SOURCE_DIR}}/{split_rel}"
                    # Only add if not already in CMakeLists
                    if split.name not in cmake_content:
                        split_entries.append(split_cmake)

                if split_entries:
                    # Insert after the original line
                    insert_text = comment + '\n' + '\n'.join(split_entries)
                    cmake_lines.insert(i + 1, insert_text)
                    found = True
                    print(f"  CMAKE: Added {len(split_entries)} entries after {pair_original.name}")
                break

        if not found and splits:
            # Couldn't find reference - just note it
            print(f"  CMAKE WARNING: Couldn't find {pair_original.name} in CMakeLists.txt")
            print(f"    Split files that need adding: {[s.name for s in splits]}")

    if not dry_run:
        new_cmake = '\n'.join(cmake_lines)
        with open(CMAKE_FILE, 'w') as f:
            f.write(new_cmake)
        print(f"\n  Updated {CMAKE_FILE}")

    return cmake_lines


def main():
    dry_run = '--dry-run' in sys.argv

    print("Finding facade pairs...")
    pairs = find_facade_pairs()
    print(f"Found {len(pairs)} original/facade pairs\n")

    if not pairs:
        print("No facades to apply.")
        return

    # Step 1: Replace originals with facades
    print("Step 1: Replacing originals with facades")
    print("-" * 50)
    for original, facade in pairs:
        original_backup = original.parent / (original.stem + ".c.orig")

        orig_lines = sum(1 for _ in open(original))
        facade_lines = sum(1 for _ in open(facade))

        print(f"  {original.name}: {orig_lines} -> {facade_lines} lines (facade)")

        if not dry_run:
            # Backup original
            shutil.copy2(original, original_backup)
            # Replace with facade
            shutil.copy2(facade, original)

    # Step 2: Update CMakeLists.txt
    print(f"\nStep 2: Updating CMakeLists.txt")
    print("-" * 50)
    update_cmake(pairs, dry_run=dry_run)

    # Step 3: Summary
    print(f"\nStep 3: Summary")
    print("-" * 50)
    total_split_files = 0
    for original, facade in pairs:
        splits = find_split_files(original)
        total_split_files += len(splits)

    print(f"  Originals replaced: {len(pairs)}")
    print(f"  Split files created: {total_split_files}")
    print(f"  {'DRY RUN - no files modified' if dry_run else 'All files modified'}")

    if not dry_run:
        # Clean up facade files (they've been applied)
        print(f"\nStep 4: Cleaning up facade files")
        for original, facade in pairs:
            os.remove(facade)
            print(f"  Removed: {facade.name}")

    print(f"\nNext: cd build && cmake .. && make nimcp -j4")


if __name__ == '__main__':
    main()
