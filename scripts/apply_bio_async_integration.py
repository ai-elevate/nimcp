#!/usr/bin/env python3
"""
Bio-Async and Logging Integration Script
Automatically integrates bio-async communication and comprehensive logging into NIMCP modules
"""

import os
import re
from pathlib import Path

# Module configuration
MODULES = {
    # Glial modules
    "src/glial/integration/nimcp_glial_integration.c": {
        "log_module": "GLIAL_INTEGRATION",
        "bio_module_id": "BIO_MODULE_GLIAL_INTEGRATION",
        "header": "include/glial/integration/nimcp_glial_integration.h"
    },
    "src/glial/astrocytes/nimcp_astrocyte_calcium.c": {
        "log_module": "ASTROCYTE_CALCIUM",
        "bio_module_id": "BIO_MODULE_ASTROCYTE",
        "header": "include/glial/astrocytes/nimcp_astrocyte_calcium.h"
    },
    "src/glial/astrocyte_types/nimcp_astrocyte_types.c": {
        "log_module": "ASTROCYTE_TYPES",
        "bio_module_id": "BIO_MODULE_ASTROCYTE",
        "header": "include/glial/astrocyte_types/nimcp_astrocyte_types.h"
    },
    "src/glial/myelin_sheath/nimcp_myelin_math.c": {
        "log_module": "MYELIN_MATH",
        "bio_module_id": "BIO_MODULE_MYELIN",
        "header": "include/glial/myelin_sheath/nimcp_myelin_math.h"
    },

    # Security modules
    "src/security/nimcp_security.c": {
        "log_module": "SECURITY",
        "bio_module_id": "BIO_MODULE_SECURITY",
        "header": "include/security/nimcp_security.h"
    },
    "src/security/nimcp_capability.c": {
        "log_module": "CAPABILITY",
        "bio_module_id": "BIO_MODULE_CAPABILITY",
        "header": "include/security/nimcp_capability.h"
    },
    "src/security/nimcp_cfi.c": {
        "log_module": "CFI",
        "bio_module_id": "BIO_MODULE_CFI",
        "header": "include/security/nimcp_cfi.h"
    },
    "src/security/nimcp_security_audit.c": {
        "log_module": "SECURITY_AUDIT",
        "bio_module_id": "BIO_MODULE_SECURITY_AUDIT",
        "header": "include/security/nimcp_security_audit.h"
    },
    "src/security/nimcp_continuous_monitor.c": {
        "log_module": "CONTINUOUS_MONITOR",
        "bio_module_id": "BIO_MODULE_CONTINUOUS_MONITOR",
        "header": "include/security/nimcp_continuous_monitor.h"
    },
    # Add more security modules...
}

BIO_ASYNC_INCLUDES = """#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
"""

def add_logging_includes(content, log_module_name):
    """Add bio-async and logging includes after first existing include"""
    # Find the last #include statement
    include_pattern = r'(#include\s+[<"][^>"]+[>"])'
    includes = list(re.finditer(include_pattern, content))

    if not includes:
        return content

    last_include = includes[-1]
    insert_pos = last_include.end()

    # Check if already has bio-async includes
    if 'bio_router.h' in content:
        print("    Already has bio-async includes")
        return content

    # Add includes and LOG_MODULE definition
    new_content = (
        content[:insert_pos] +
        "\n" + BIO_ASYNC_INCLUDES +
        f'\n#define LOG_MODULE "{log_module_name}"\n' +
        content[insert_pos:]
    )

    return new_content

def convert_logging_calls(content):
    """Convert old logging to new LOG_* macros"""
    # Replace NIMCP_LOGGING_* with LOG_*
    content = re.sub(r'NIMCP_LOGGING_DEBUG\s*\(\s*"([^"]*)"',
                     r'LOG_DEBUG(LOG_MODULE, "\1"', content)
    content = re.sub(r'NIMCP_LOGGING_INFO\s*\(\s*"([^"]*)"',
                     r'LOG_INFO(LOG_MODULE, "\1"', content)
    content = re.sub(r'NIMCP_LOGGING_WARN\s*\(\s*"([^"]*)"',
                     r'LOG_WARN(LOG_MODULE, "\1"', content)
    content = re.sub(r'NIMCP_LOGGING_ERROR\s*\(\s*"([^"]*)"',
                     r'LOG_ERROR(LOG_MODULE, "\1"', content)

    return content

def process_module(src_path, config):
    """Process a single module file"""
    if not os.path.exists(src_path):
        print(f"  [SKIP] File not found: {src_path}")
        return False

    print(f"Processing: {src_path}")

    # Read source file
    with open(src_path, 'r') as f:
        content = f.read()

    # Check if already integrated
    if 'bio_router.h' in content:
        print("  [OK] Already has bio-async integration")
        return False

    # Add includes
    content = add_logging_includes(content, config['log_module'])

    # Convert logging calls
    content = convert_logging_calls(content)

    # Write back
    # with open(src_path, 'w') as f:
    #     f.write(content)

    print(f"  [DRY-RUN] Would update: {src_path}")
    return True

def main():
    print("=" * 70)
    print("Bio-Async and Logging Integration")
    print("=" * 70)
    print()

    modified_count = 0
    skipped_count = 0

    for src_path, config in MODULES.items():
        if process_module(src_path, config):
            modified_count += 1
        else:
            skipped_count += 1

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Files to modify: {modified_count}")
    print(f"Files skipped: {skipped_count}")
    print()
    print("NOTE: This is a dry-run. Uncomment write code to apply changes.")

if __name__ == "__main__":
    main()
