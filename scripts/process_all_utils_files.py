#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

def find_util_files(base_dir):
    """Find all C files in utils directory"""
    utils_dir = Path(base_dir) / 'src' / 'utils'
    c_files = list(utils_dir.rglob('*.c'))
    return sorted(c_files)

def process_file(file_path, script_path):
    """Process a single file using the integration script"""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, '', str(e)

def main():
    base_dir = Path('/home/bbrelin/nimcp')
    script_path = base_dir / 'scripts' / 'integrate_all_utils.py'

    print("="*70)
    print("Utils Integration - Logging & Unified Memory")
    print("="*70)
    print()

    # Find all files
    files = find_util_files(base_dir)
    print(f"Found {len(files)} C files in utils/\n")

    # Track stats
    stats = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'malloc': 0,
        'calloc': 0,
        'realloc': 0,
        'free': 0
    }

    module_stats = defaultdict(lambda: {'files': 0, 'malloc': 0, 'calloc': 0, 'realloc': 0, 'free': 0})

    # Process each file
    for file_path in files:
        stats['total'] += 1

        # Get module name
        module = file_path.parts[-2]  # Parent directory name

        success, stdout, stderr = process_file(file_path, script_path)

        if not success:
            stats['errors'] += 1
            print(f"ERROR: {file_path.name}")
            if stderr:
                print(f"  {stderr[:200]}")
            continue

        if 'SKIP' in stdout:
            stats['skipped'] += 1
        else:
            stats['processed'] += 1
            module_stats[module]['files'] += 1

            # Extract memory replacement counts
            import re
            malloc_match = re.search(r'malloc:(\d+)', stdout)
            calloc_match = re.search(r'calloc:(\d+)', stdout)
            realloc_match = re.search(r'realloc:(\d+)', stdout)
            free_match = re.search(r'free:(\d+)', stdout)

            if malloc_match:
                count = int(malloc_match.group(1))
                stats['malloc'] += count
                module_stats[module]['malloc'] += count

            if calloc_match:
                count = int(calloc_match.group(1))
                stats['calloc'] += count
                module_stats[module]['calloc'] += count

            if realloc_match:
                count = int(realloc_match.group(1))
                stats['realloc'] += count
                module_stats[module]['realloc'] += count

            if free_match:
                count = int(free_match.group(1))
                stats['free'] += count
                module_stats[module]['free'] += count

        print(stdout)

    # Print summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files:      {stats['total']}")
    print(f"Processed:        {stats['processed']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Errors:           {stats['errors']}")
    print()
    print("Memory Replacements:")
    print(f"  malloc  -> nimcp_malloc:  {stats['malloc']}")
    print(f"  calloc  -> nimcp_calloc:  {stats['calloc']}")
    print(f"  realloc -> nimcp_realloc: {stats['realloc']}")
    print(f"  free    -> nimcp_free:    {stats['free']}")
    print(f"  TOTAL:                    {stats['malloc'] + stats['calloc'] + stats['realloc'] + stats['free']}")

    # Module breakdown
    print()
    print("="*70)
    print("MODULE BREAKDOWN")
    print("="*70)
    for module in sorted(module_stats.keys()):
        ms = module_stats[module]
        total = ms['malloc'] + ms['calloc'] + ms['realloc'] + ms['free']
        print(f"\n{module}:")
        print(f"  Files:    {ms['files']}")
        if total > 0:
            print(f"  malloc:   {ms['malloc']}")
            print(f"  calloc:   {ms['calloc']}")
            print(f"  realloc:  {ms['realloc']}")
            print(f"  free:     {ms['free']}")
            print(f"  Total:    {total}")
        else:
            print(f"  (no memory calls)")

if __name__ == '__main__':
    main()
