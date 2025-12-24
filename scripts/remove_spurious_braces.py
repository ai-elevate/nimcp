#!/usr/bin/env python3
"""
Remove spurious `};` that were incorrectly added.

The pattern: standalone `};` that appears after:
1. A proper typedef struct close like `} typename;`
2. A function declaration ending in `;`
3. After another `};`
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def remove_spurious_braces(filepath):
    """Remove spurious standalone };"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: }; on its own line after a proper typedef struct close
    # Like: } bridge_config_base_t;\n\n};\n
    pattern1 = r'(\}\s+\w+_t\s*;)\s*\n\n\};'

    # Pattern 2: }; after a function declaration (ends with )\s*;)
    pattern2 = r'(\)\s*;)\s*\n\n\};'

    # Pattern 3: double }; (one was already there)
    pattern3 = r'\};\s*\n\n\};'

    # Pattern 4: standalone }; at end of file or before comments that aren't struct related
    pattern4 = r'(\)\s*;)\s*\n\};(\n+/\* =)'

    original = content

    # Apply removals
    content = re.sub(pattern1, r'\1', content)
    content = re.sub(pattern2, r'\1', content)
    content = re.sub(pattern3, r'};', content)
    content = re.sub(pattern4, r'\1\2', content)

    # Also remove standalone }; that appear after #endif or before #ifdef
    content = re.sub(r'\n\};(\n+#endif)', r'\1', content)
    content = re.sub(r'(\n#endif[^\n]*)\n\n\};', r'\1', content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("Removing spurious }; that were incorrectly added")
    print("=" * 60)

    fixed_count = 0

    # Process headers
    include_dir = PROJECT_ROOT / 'include'
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            filepath = Path(root) / file
            if remove_spurious_braces(filepath):
                print(f"Fixed: {filepath.relative_to(PROJECT_ROOT)}")
                fixed_count += 1

    print(f"\nFixed {fixed_count} headers")

if __name__ == '__main__':
    main()
