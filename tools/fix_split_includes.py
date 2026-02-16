#!/usr/bin/env python3
"""
Fix Split File Includes: Replace internal header include with original file's includes
======================================================================================
Instead of using our auto-generated internal headers (which miss dependencies),
each split file should include exactly the same headers as the original .c file.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"


def get_original_includes(orig_path):
    """Extract all include lines and top-level declarations from original .c file."""
    if not orig_path.exists():
        return None

    with open(orig_path, 'r') as f:
        lines = f.readlines()

    includes = []
    other_top = []

    # Read until we hit the first function definition or struct definition
    in_comment = False
    for line in lines:
        stripped = line.strip()

        # Track multi-line comments
        if '/*' in stripped:
            in_comment = True
        if '*/' in stripped:
            in_comment = False
            continue
        if in_comment:
            continue

        # Stop at first function definition
        if re.match(r'^(static\s+)?(inline\s+)?(void|int|bool|float|double|char|uint|nimcp_|size_t|ssize_t|brain_|omni_|nlp_|rcog_|swarm_|working_|global_|mirror_|salience_|knowledge_|ethics_|wellbeing_|introspection_|plasticity_|language_|portia_|distrib_|collective_|hypergraph_|lnn_|fep_|neuromod_|systems_|bio_|health_agent|checkpoint_|deadlock_|runtime_|logging_)', stripped):
            if '(' in stripped and not stripped.startswith('#'):
                break

        if stripped.startswith('#include'):
            includes.append(line.rstrip())
        elif stripped.startswith('#define') and not stripped.endswith('_H') and not stripped.endswith('_H_'):
            other_top.append(line.rstrip())

    return includes


def fix_split_file(split_path, orig_includes):
    """Replace internal header include with original includes in a split file."""
    with open(split_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    replaced = False

    for line in lines:
        if '_internal.h"' in line and line.strip().startswith('#include'):
            if not replaced:
                # Replace with original includes
                new_lines.append("/* Original includes (from pre-split source) */")
                for inc in orig_includes:
                    new_lines.append(inc)
                replaced = True
            # Skip the internal header include
        else:
            new_lines.append(line)

    if replaced:
        with open(split_path, 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    return False


def main():
    dry_run = '--dry-run' in os.sys.argv

    # Find all .c.orig files (backups of originals)
    orig_files = sorted(SRC_ROOT.rglob("*.c.orig"))
    fixed = 0
    total_splits = 0

    for orig in orig_files:
        module_name = orig.stem.replace('.c', '')  # e.g., nimcp_brain_immune
        parent = orig.parent

        # Get original includes
        orig_includes = get_original_includes(orig)
        if not orig_includes:
            continue

        # Find all split files for this module
        split_files = []
        for f in sorted(parent.iterdir()):
            if f.is_file() and f.suffix == '.c' and f.stem.startswith(module_name + '_') and \
               not f.stem.endswith('_facade') and f.name != orig.name:
                split_files.append(f)

        if not split_files:
            continue

        print(f"\n{module_name}: {len(split_files)} split files, {len(orig_includes)} includes")

        for sf in split_files:
            total_splits += 1
            if not dry_run:
                if fix_split_file(sf, orig_includes):
                    print(f"  Fixed: {sf.name}")
                    fixed += 1
                else:
                    print(f"  No internal header in: {sf.name}")
            else:
                print(f"  Would fix: {sf.name}")
                fixed += 1

    # Also fix the facade files (the replaced originals)
    for orig in orig_files:
        module_name = orig.stem.replace('.c', '')
        facade = orig.parent / (module_name + '.c')
        if facade.exists():
            orig_includes = get_original_includes(orig)
            if orig_includes and not dry_run:
                if fix_split_file(facade, orig_includes):
                    print(f"  Fixed facade: {facade.name}")
                    fixed += 1

    print(f"\n{'='*50}")
    print(f"Fixed {fixed}/{total_splits} split files")


if __name__ == '__main__':
    main()
