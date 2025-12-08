#!/usr/bin/env python3
"""
Script to integrate bio-async, logging, and security into training modules.
This script automates the integration process for all training modules.
"""

import os
import re
import sys

# File paths
FILES = {
    'loss_functions': '/home/bbrelin/nimcp/src/middleware/training/nimcp_loss_functions.c',
    'lr_scheduler': '/home/bbrelin/nimcp/src/middleware/training/nimcp_lr_scheduler.c',
    'gradient_manager': '/home/bbrelin/nimcp/src/middleware/training/nimcp_gradient_manager.c',
    'training_callbacks': '/home/bbrelin/nimcp/src/middleware/training/nimcp_training_callbacks.c',
    'training_module': '/home/bbrelin/nimcp/src/middleware/training/nimcp_training_module.c',
}

def add_headers(filepath, module_name):
    """Add bio-async, logging, and security headers to a C file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already integrated
    if 'async/nimcp_bio_async.h' in content:
        print(f"  {module_name}: Already integrated, skipping")
        return False

    # Find the last #include before the first #define or code
    pattern = r'(#include\s+["\<][^\>"\n]+["\>]\s*\n)+((?:#define|/\*\*|typedef|struct|static))'
    match = re.search(pattern, content)

    if match:
        insertion_point = match.start(2)
        headers = '''#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

'''
        content = content[:insertion_point] + headers + content[insertion_point:]

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  {module_name}: Added headers")
        return True

    print(f"  {module_name}: Could not find insertion point")
    return False

def add_log_module_define(filepath, module_name):
    """Add LOG_MODULE define near the top of the file."""
    with open(filepath, 'r') as f:
        content = f.read()

    if '#define LOG_MODULE' in content:
        return False

    # Find first #define line
    pattern = r'(#define\s+\w+\s+.*\n)'
    match = re.search(pattern, content)

    if match:
        insertion_point = match.end()
        log_define = f'#define LOG_MODULE "{module_name}"\n'
        content = content[:insertion_point] + log_define + content[insertion_point:]

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  {module_name}: Added LOG_MODULE define")
        return True

    return False

def main():
    print("Integrating bio-async, logging, and security into training modules...")
    print()

    for module_name, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"  {module_name}: File not found - {filepath}")
            continue

        print(f"Processing {module_name}...")
        add_headers(filepath, module_name)
        add_log_module_define(filepath, module_name)
        print()

    print("Integration complete!")
    print()
    print("Note: Manual integration still required for:")
    print("  - Adding bio_ctx fields to context structs")
    print("  - Implementing message handlers")
    print("  - Adding logging calls in key functions")
    print("  - Adding BBB validation in critical paths")

if __name__ == '__main__':
    main()
