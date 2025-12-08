#!/usr/bin/env python3
"""
Script to integrate bio-async, logging, and security into middleware buffering modules.
"""

import re
import sys

def integrate_sliding_window():
    """Integrate bio-async and security into sliding_window.c"""
    file_path = "/home/bbrelin/nimcp/src/middleware/buffering/nimcp_sliding_window.c"

    with open(file_path, 'r') as f:
        content = f.read()

    # Add BIO_MODULE define
    content = content.replace(
        '#define LOG_MODULE_ID 0x0513',
        '#define LOG_MODULE_ID 0x0513\n#define BIO_MODULE_SLIDING_WINDOW 0x0513'
    )

    # Add security include
    if 'security/nimcp_blood_brain_barrier.h' not in content:
        content = content.replace(
            '#include "utils/logging/nimcp_logging.h"',
            '#include "utils/logging/nimcp_logging.h"\n#include "security/nimcp_blood_brain_barrier.h"'
        )

    # Add bio-async fields to struct
    struct_pattern = r'(struct sliding_window \{[^}]+)(    // Data storage[^}]+\})'
    if re.search(struct_pattern, content):
        content = re.sub(
            r'(    // Memory pool for temporary allocations.*?memory_pool_t temp_buffer_pool;.*?)',
            r'\1\n\n    // Bio-async integration\n    bio_module_context_t bio_ctx;\n    bool bio_async_enabled;\n\n    // Security\n    bbb_input_gate_t input_gate;',
            content, flags=re.DOTALL
        )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated {file_path}")

def integrate_temporal_accumulator():
    """Integrate bio-async and security into temporal_accumulator.c"""
    file_path = "/home/bbrelin/nimcp/src/middleware/buffering/nimcp_temporal_accumulator.c"

    with open(file_path, 'r') as f:
        content = f.read()

    # Add BIO_MODULE define
    content = content.replace(
        '#define LOG_MODULE_ID 0x0514',
        '#define LOG_MODULE_ID 0x0514\n#define BIO_MODULE_TEMPORAL_ACCUMULATOR 0x0514'
    )

    # Add security include
    if 'security/nimcp_blood_brain_barrier.h' not in content:
        content = content.replace(
            '#include "utils/logging/nimcp_logging.h"',
            '#include "utils/logging/nimcp_logging.h"\n#include "security/nimcp_blood_brain_barrier.h"'
        )

    # Add bio-async fields to struct
    struct_pattern = r'(struct temporal_accumulator \{[^}]+channel_state_t\* channels;.*?)'
    if re.search(struct_pattern, content, re.DOTALL):
        content = re.sub(
            r'(    channel_state_t\* channels;.*?\};)',
            r'    channel_state_t* channels;\n\n    // Bio-async integration\n    bio_module_context_t bio_ctx;\n    bool bio_async_enabled;\n\n    // Security\n    bbb_input_gate_t input_gate;\n};',
            content, flags=re.DOTALL
        )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated {file_path}")

def integrate_integration_buffer():
    """Integrate bio-async and security into integration_buffer.c"""
    file_path = "/home/bbrelin/nimcp/src/middleware/buffering/nimcp_integration_buffer.c"

    with open(file_path, 'r') as f:
        content = f.read()

    # Add BIO_MODULE define
    content = content.replace(
        '#define LOG_MODULE_ID 0x0511',
        '#define LOG_MODULE_ID 0x0511\n#define BIO_MODULE_INTEGRATION_BUFFER 0x0511'
    )

    # Add security include
    if 'security/nimcp_blood_brain_barrier.h' not in content:
        content = content.replace(
            '#include "utils/logging/nimcp_logging.h"',
            '#include "utils/logging/nimcp_logging.h"\n#include "security/nimcp_blood_brain_barrier.h"'
        )

    # Add bio-async fields to struct
    struct_pattern = r'(struct integration_buffer \{[^}]+channel_buffers_t\* channels;.*?)'
    if re.search(struct_pattern, content, re.DOTALL):
        content = re.sub(
            r'(    channel_buffers_t\* channels;.*?\};)',
            r'    channel_buffers_t* channels;\n\n    // Bio-async integration\n    bio_module_context_t bio_ctx;\n    bool bio_async_enabled;\n\n    // Security\n    bbb_input_gate_t input_gate;\n};',
            content, flags=re.DOTALL
        )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Updated {file_path}")

def main():
    print("Integrating bio-async and security into buffering modules...")
    print()

    try:
        integrate_sliding_window()
        integrate_temporal_accumulator()
        integrate_integration_buffer()

        print()
        print("✓ All buffering modules updated successfully")
        print()
        print("Note: You still need to:")
        print("  1. Add bio-async initialization in create functions")
        print("  2. Add bio-async cleanup in destroy functions")
        print("  3. Add broadcast calls for key events")
        print("  4. Add message handlers")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
