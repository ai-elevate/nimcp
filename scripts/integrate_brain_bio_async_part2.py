#!/usr/bin/env python3
"""
Integrate bio-async, logging, and unified memory into remaining brain modules
"""

import re
import sys
from pathlib import Path

def add_includes(content, log_module):
    """Add bio-async, logging, and unified memory includes"""

    # Check if already has bio-async
    if 'nimcp_bio_async.h' in content:
        print("  - Already has bio-async")
        has_bio_async = True
    else:
        has_bio_async = False

    # Check if already has logging
    if 'nimcp_logging.h' in content:
        print("  - Already has logging")
        has_logging = True
    else:
        has_logging = False

    # Check if already has LOG_MODULE
    if '#define LOG_MODULE' in content:
        print("  - Already has LOG_MODULE")
        has_log_module = True
    else:
        has_log_module = False

    # If everything is there, return
    if has_bio_async and has_logging and has_log_module:
        print("  - File already fully integrated")
        return content

    # Find first #include statement
    include_pattern = r'^(#include\s+[<"])'
    match = re.search(include_pattern, content, re.MULTILINE)

    if not match:
        print("  Warning: No includes found")
        return content

    # Build includes to add
    includes_to_add = []

    if not has_bio_async:
        includes_to_add.append("""// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
""")
        print("  + Adding bio-async includes")

    if not has_logging:
        includes_to_add.append("""// Logging integration
#include "utils/logging/nimcp_logging.h"
""")
        print("  + Adding logging include")

    # Don't add unified memory if it's already using nimcp_malloc
    if 'nimcp_malloc' not in content and 'nimcp_unified_memory.h' not in content:
        includes_to_add.append("""// Unified memory integration
#include "utils/memory/nimcp_unified_memory.h"
""")
        print("  + Adding unified memory include")

    if includes_to_add:
        # Insert our includes before the first include
        pos = match.start()
        new_includes = '\n'.join(includes_to_add) + '\n'
        content = content[:pos] + new_includes + content[pos:]

    # Add LOG_MODULE if not present
    if not has_log_module:
        # Find end of includes
        last_include = list(re.finditer(r'^#include.*$', content, re.MULTILINE))[-1]
        insert_pos = last_include.end()

        # Find next non-empty line
        next_line_match = re.search(r'\n\s*\n', content[insert_pos:])
        if next_line_match:
            insert_pos += next_line_match.start() + 1
        else:
            insert_pos += 1

        log_define = f'\n#define LOG_MODULE "{log_module}"\n'
        content = content[:insert_pos] + log_define + content[insert_pos:]
        print(f"  + Adding LOG_MODULE={log_module}")

    return content

def process_file(filepath, log_module):
    """Process a single brain module file"""

    print(f"\nProcessing {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Add includes
    content = add_includes(content, log_module)

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Integrated {filepath}")

def main():
    # Define remaining files to process
    files_to_process = [
        {
            'path': 'src/core/brain/oscillations/nimcp_brain_complex_oscillations.c',
            'log_module': 'BRAIN_OSCILLATIONS',
        },
        {
            'path': 'src/core/brain/analysis/nimcp_brain_topology.c',
            'log_module': 'BRAIN_TOPOLOGY',
        },
        {
            'path': 'src/core/brain/distributed/nimcp_brain_distributed.c',
            'log_module': 'BRAIN_DISTRIBUTED',
        },
        {
            'path': 'src/core/brain/strategy/nimcp_brain_strategy.c',
            'log_module': 'BRAIN_STRATEGY',
        },
        {
            'path': 'src/core/brain/factory/nimcp_brain_factory.c',
            'log_module': 'BRAIN_FACTORY',
        },
        {
            'path': 'src/core/brain/persistence/nimcp_brain_persistence.c',
            'log_module': 'BRAIN_PERSIST',
        },
        {
            'path': 'src/core/brain/inference/nimcp_brain_inference.c',
            'log_module': 'BRAIN_INFERENCE',
        },
    ]

    base_path = Path('/home/bbrelin/nimcp')

    for file_info in files_to_process:
        filepath = base_path / file_info['path']
        if filepath.exists():
            process_file(filepath, file_info['log_module'])
        else:
            print(f"Warning: File not found: {filepath}")

    print("\n" + "="*70)
    print("All files integrated successfully!")
    print("="*70)

if __name__ == '__main__':
    main()
