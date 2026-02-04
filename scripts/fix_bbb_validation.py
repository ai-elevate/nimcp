#!/usr/bin/env python3
"""Add BBB validation to external API entry points."""
import os, re

BASE = "/home/bbrelin/nimcp"

def add_include(filepath, include_str):
    """Add an include to a file if not already present."""
    with open(filepath, 'r') as f:
        content = f.read()
    if include_str in content:
        return False
    # Find last #include line with nimcp or security
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#include') and ('nimcp' in line or 'security' in line):
            insert_idx = i + 1
    if insert_idx > 0:
        lines.insert(insert_idx, include_str)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return True
    return False

def add_bbb_check_after_null_check(filepath, func_pattern, param_name, check_type, max_len=None):
    """Add BBB check after an existing NULL check for a parameter."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if f'bbb_check_{check_type}({param_name}' in content:
        return False  # Already has this check
    
    # Find the function and the NULL check
    if check_type == 'string':
        bbb_call = f'    if (!bbb_check_string({param_name}, {max_len}, "{func_pattern}")) {{\n        return NIMCP_ERROR;\n    }}\n'
    else:
        bbb_call = f'    if (!bbb_check_pointer({param_name}, "{func_pattern}")) {{\n        return NIMCP_ERROR;\n    }}\n'
    
    # Find the pattern: NIMCP_CHECK_THROW(param, ...) or if (!param) { ... return ...}
    # Insert BBB check after it
    null_patterns = [
        f'NIMCP_CHECK_THROW({param_name},',
        f'NIMCP_API_CHECK_NULL({param_name},',
        f'if (!{param_name})',
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for pat in null_patterns:
            if pat in line:
                # Find end of this check block
                j = i + 1
                while j < len(lines) and ('}' not in lines[j] and 'return' not in lines[j]):
                    j += 1
                if j < len(lines) and ('return' in lines[j] or '}' in lines[j]):
                    # Find the next blank line or statement
                    k = j + 1
                    while k < len(lines) and lines[k].strip() == '':
                        k += 1
                    # Insert BBB check
                    lines.insert(k, '')
                    lines.insert(k + 1, f'    // BBB: Validate external {check_type} input')
                    for idx, bbb_line in enumerate(bbb_call.rstrip().split('\n')):
                        lines.insert(k + 2 + idx, bbb_line)
                    with open(filepath, 'w') as f:
                        f.write('\n'.join(lines))
                    return True
    return False

def main():
    print("=" * 70)
    print("NIMCP BBB Validation - API Boundary Enforcement")
    print("=" * 70)
    
    total = 0
    
    # Add bbb_helpers.h include to API files
    api_files = [
        "src/api/nimcp.c",
        "src/api/nimcp_api_brain.c",
        "src/api/nimcp_api_inference.c",
        "src/api/nimcp_api_cognitive.c",
        "src/api/nimcp_api_network.c",
        "src/api/nimcp_api_training.c",
        "src/api/nimcp_subsystems_api.c",
        "src/api/nimcp_refactored.c",
        "src/bindings/python/nimcp_python.c",
    ]
    
    for rel_path in api_files:
        filepath = os.path.join(BASE, rel_path)
        if not os.path.exists(filepath):
            print(f"  SKIP: {rel_path} not found")
            continue
        if add_include(filepath, '#include "security/nimcp_bbb_helpers.h"'):
            total += 1
            print(f"  [INCLUDE] {rel_path}")
    
    # Add bbb_helpers_init() to nimcp.c nimcp_init function
    nimcp_c = os.path.join(BASE, "src/api/nimcp.c")
    with open(nimcp_c, 'r') as f:
        content = f.read()
    
    if 'bbb_helpers_init' not in content:
        # Find nimcp_init function and add bbb_helpers_init after existing init calls
        content = content.replace(
            '    // Initialize exception system',
            '    // Initialize BBB helpers\n    bbb_helpers_init();\n    bbb_register_module("nimcp_api", BBB_MODULE_TYPE_CORE);\n\n    // Initialize exception system',
            1
        )
        if 'bbb_helpers_init' in content:
            with open(nimcp_c, 'w') as f:
                f.write(content)
            total += 1
            print(f"  [INIT] src/api/nimcp.c - added bbb_helpers_init()")
    
    # Add bbb_helpers_shutdown() to nimcp.c nimcp_shutdown
    with open(nimcp_c, 'r') as f:
        content = f.read()
    
    if 'bbb_helpers_shutdown' not in content:
        content = content.replace(
            '    nimcp_exception_system_shutdown();',
            '    bbb_helpers_shutdown();\n    nimcp_exception_system_shutdown();',
            1
        )
        if 'bbb_helpers_shutdown' in content:
            with open(nimcp_c, 'w') as f:
                f.write(content)
            total += 1
            print(f"  [SHUTDOWN] src/api/nimcp.c - added bbb_helpers_shutdown()")
    
    print(f"\nTotal changes: {total}")
    print("=" * 70)

if __name__ == '__main__':
    main()
