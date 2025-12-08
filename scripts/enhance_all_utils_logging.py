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

def enhance_file(file_path, script_path):
    """Enhance a single file with logging statements"""
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
    script_path = base_dir / 'scripts' / 'add_logging_statements.py'

    print("="*70)
    print("Utils Logging Enhancement")
    print("="*70)
    print()

    # Find all files
    files = find_util_files(base_dir)
    print(f"Found {len(files)} C files in utils/\n")

    # Track stats
    stats = {
        'total': 0,
        'enhanced': 0,
        'skipped': 0,
        'errors': 0,
        'logs_added': 0
    }

    module_stats = defaultdict(lambda: {'files': 0, 'logs': 0})

    # Process each file
    for file_path in files:
        stats['total'] += 1

        # Get module name
        module = file_path.parts[-2]  # Parent directory name

        success, stdout, stderr = enhance_file(file_path, script_path)

        if not success:
            stats['errors'] += 1
            print(f"ERROR: {file_path.name}")
            if stderr:
                print(f"  {stderr[:200]}")
            continue

        if 'SKIP' in stdout:
            stats['skipped'] += 1
        else:
            stats['enhanced'] += 1
            module_stats[module]['files'] += 1

            # Extract log count
            import re
            log_match = re.search(r'added (\d+) log', stdout)
            if log_match:
                count = int(log_match.group(1))
                stats['logs_added'] += count
                module_stats[module]['logs'] += count

        print(stdout)

    # Print summary
    print()
    print("="*70)
    print("LOGGING ENHANCEMENT SUMMARY")
    print("="*70)
    print(f"Total files:       {stats['total']}")
    print(f"Enhanced:          {stats['enhanced']}")
    print(f"Skipped:           {stats['skipped']}")
    print(f"Errors:            {stats['errors']}")
    print(f"Log statements:    {stats['logs_added']}")

    # Module breakdown
    print()
    print("="*70)
    print("MODULE BREAKDOWN")
    print("="*70)
    for module in sorted(module_stats.keys()):
        ms = module_stats[module]
        print(f"{module}:")
        print(f"  Files:       {ms['files']}")
        print(f"  Logs added:  {ms['logs']}")

if __name__ == '__main__':
    main()
