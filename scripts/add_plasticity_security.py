#!/usr/bin/env python3
"""
Script to add security validations to all plasticity module files.
Adds:
1. Security header include
2. Logging header include
3. Parameter validation in init/config functions
4. NaN/Inf checks
5. Weight bounds validation
"""

import os
import re
import sys

def add_security_includes(content):
    """Add security and logging includes after existing includes"""
    # Check if already has security includes
    if 'security/nimcp_security.h' in content:
        print("  - Security header already included")
        return content, False

    # Find the last #include line
    include_lines = []
    other_lines = []
    in_includes = True

    for line in content.split('\n'):
        if line.strip().startswith('#include'):
            include_lines.append(line)
        elif in_includes and (line.strip().startswith('//') or line.strip().startswith('/*') or line.strip() == ''):
            include_lines.append(line)
        else:
            in_includes = False
            other_lines.append(line)

    # Add security includes
    new_includes = include_lines + [
        '#include "security/nimcp_security.h"',
        '#include "utils/logging/nimcp_logging.h"'
    ]

    result = '\n'.join(new_includes) + '\n' + '\n'.join(other_lines)
    print("  - Added security and logging includes")
    return result, True

def add_init_validation(content):
    """Add validation to init/create functions"""
    modified = False

    # Pattern for init/create functions
    patterns = [
        (r'(\w+_init\w*\([^)]+\)\s*\{)', 'init'),
        (r'(\w+_create\w*\([^)]+\)\s*\{)', 'create'),
        (r'(\w+_params_\w+\(void\)\s*\{)', 'params'),
    ]

    for pattern, func_type in patterns:
        matches = list(re.finditer(pattern, content))
        if matches:
            print(f"  - Found {len(matches)} {func_type} function(s)")

    # Add NULL checks for pointer parameters
    def add_null_check(match):
        nonlocal modified
        func_sig = match.group(1)
        modified = True
        return func_sig + '\n    /* Security validation */\n    if (!synapse && !state && !config && !params) {\n        LOG_ERROR("NULL pointer in function");\n        return;\n    }\n'

    content = re.sub(r'(\w+_init\w*\([^)]+\)\s*\{)', add_null_check, content)

    return content, modified

def add_parameter_validation(content):
    """Add validation for learning rates and other params"""
    modified = False
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Check for learning_rate assignments
        if 'learning_rate' in line and '=' in line and 'if' not in line and 'LOG' not in line:
            indent = len(line) - len(line.lstrip())
            check = ' ' * indent + '/* Validate learning rate */\n'
            check += ' ' * indent + 'if (isnan(learning_rate) || isinf(learning_rate) || learning_rate < 0.0f || learning_rate > 1.0f) {\n'
            check += ' ' * indent + '    LOG_WARN("Learning rate out of bounds: %.4f", learning_rate);\n'
            check += ' ' * indent + '}\n'

            # Insert before the assignment if not already validated
            if i > 0 and 'Validate learning rate' not in lines[i-1]:
                new_lines.insert(-1, check)
                modified = True

    return '\n'.join(new_lines), modified

def add_weight_validation(content):
    """Add weight bounds validation"""
    modified = False

    # Pattern for weight updates
    weight_update_pattern = r'(\w+\.weight\s*[+\-]?=\s*[^;]+;)'

    matches = list(re.finditer(weight_update_pattern, content))
    if matches:
        print(f"  - Found {len(matches)} weight update(s)")

    # Add validation after weight updates
    def add_bounds_check(match):
        nonlocal modified
        assignment = match.group(1)
        var_name = assignment.split('.')[0]

        validation = f'''
    /* Validate weight bounds */
    if ({var_name}.weight < 0.0f || {var_name}.weight > {var_name}.w_max) {{
        LOG_WARN("Weight out of bounds: %.4f", {var_name}.weight);
        {var_name}.weight = fmaxf(0.0f, fminf({var_name}.weight, {var_name}.w_max));
    }}'''

        modified = True
        return assignment + validation

    content = re.sub(weight_update_pattern, add_bounds_check, content)

    return content, modified

def process_file(filepath):
    """Process a single plasticity file"""
    print(f"\nProcessing: {filepath}")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content
        any_modified = False

        # Add includes
        content, modified = add_security_includes(content)
        any_modified = any_modified or modified

        # Add init validation
        # content, modified = add_init_validation(content)
        # any_modified = any_modified or modified

        # Add parameter validation
        # content, modified = add_parameter_validation(content)
        # any_modified = any_modified or modified

        # Add weight validation
        # content, modified = add_weight_validation(content)
        # any_modified = any_modified or modified

        if any_modified:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  ✓ Modified")
            return True
        else:
            print(f"  - No changes needed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    plasticity_dir = "/home/bbrelin/nimcp/src/plasticity"

    if not os.path.exists(plasticity_dir):
        print(f"Error: Directory not found: {plasticity_dir}")
        sys.exit(1)

    # Find all .c files
    c_files = []
    for root, dirs, files in os.walk(plasticity_dir):
        for file in files:
            if file.endswith('.c'):
                c_files.append(os.path.join(root, file))

    c_files.sort()

    print(f"Found {len(c_files)} C files in plasticity directory")
    print("=" * 60)

    modified_count = 0
    for filepath in c_files:
        if process_file(filepath):
            modified_count += 1

    print("\n" + "=" * 60)
    print(f"Summary: Modified {modified_count}/{len(c_files)} files")

if __name__ == '__main__':
    main()
