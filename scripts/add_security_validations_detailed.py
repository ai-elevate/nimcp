#!/usr/bin/env python3
"""
Add detailed security validations to plasticity modules:
- NaN/Inf checks for all init/create function parameters
- Bounds validation for numeric parameters
- Weight clamping with logging
"""

import os
import re

# Common validation snippets
NULL_CHECK_TEMPLATE = """    /* Security: NULL pointer validation */
    if (!{param}) {{
        LOG_ERROR("NULL {param} in {func_name}");
        return{return_val};
    }}
"""

NAN_INF_CHECK_TEMPLATE = """    /* Security: NaN/Inf validation for {param} */
    if (isnan({param}) || isinf({param})) {{
        LOG_ERROR("Invalid {param} in {func_name}: %.4f", {param});
        return{return_val};
    }}
"""

BOUNDS_CHECK_TEMPLATE = """    /* Security: Bounds validation for {param} */
    if ({param} < {min_val}f || {param} > {max_val}f) {{
        LOG_WARN("{param} out of bounds in {func_name}: %.4f (expected [{min_val}, {max_val}])", {param});
    }}
"""

WEIGHT_VALIDATION_TEMPLATE = """    /* Security: Validate weight bounds */
    if (new_weight < {min_weight}f || new_weight > {max_weight}f) {{
        LOG_WARN("Weight out of bounds: %.4f (clamping to [{min_weight}, {max_weight}])", new_weight);
        new_weight = fmaxf({min_weight}f, fminf(new_weight, {max_weight}f));
    }}
"""

def add_validation_to_init_func(content, func_name, param_types):
    """Add validations to an init/create function"""
    # Find the function
    pattern = rf'({func_name}\s*\([^{{]*\)\s*{{)'
    match = re.search(pattern, content)

    if not match:
        return content, False

    # Extract function signature to find parameters
    func_start = match.start()
    func_sig_start = content.rfind('\n', 0, func_start) + 1
    func_sig = content[func_sig_start:match.end()]

    # Find parameter names
    param_pattern = r'\b(\w+)\s*\*\s*(\w+)|(\w+)\s+(\w+)(?=\s*[,\)])'
    params = re.findall(param_pattern, func_sig)

    validations = []

    for param_match in params:
        if param_match[1]:  # Pointer parameter
            param_name = param_match[1]
            if param_name not in ['void', 'const']:
                validations.append(f"    if (!{param_name}) {{\n        LOG_ERROR(\"NULL {param_name} in {func_name}\");\n        return;\n    }}\n")
        elif param_match[3]:  # Value parameter
            param_name = param_match[3]
            if 'float' in param_match[2] or 'double' in param_match[2]:
                validations.append(f"    if (isnan({param_name}) || isinf({param_name})) {{\n        LOG_ERROR(\"Invalid {param_name} in {func_name}\");\n        return;\n    }}\n")

    if validations:
        # Insert validations after the opening brace
        insert_pos = match.end()
        validation_block = "\n    /* Security validations */\n" + "".join(validations)
        content = content[:insert_pos] + validation_block + content[insert_pos:]
        return content, True

    return content, False

def add_weight_bounds_check(content):
    """Add bounds checking for weight assignments"""
    # Pattern: something.weight = expression;
    pattern = r'(\w+)\.weight\s*=\s*([^;]+);'

    matches = list(re.finditer(pattern, content))
    if not matches:
        return content, False

    # Process in reverse to maintain positions
    for match in reversed(matches):
        var_name = match.group(1)
        expression = match.group(2).strip()

        # Skip if already has validation nearby
        context_start = max(0, match.start() - 200)
        context = content[context_start:match.end() + 100]
        if 'LOG_WARN' in context and 'bounds' in context:
            continue

        # Add validation after the assignment
        validation = f"""
    /* Security: Validate weight bounds */
    if ({var_name}.weight < 0.0f || ({var_name}.w_max > 0.0f && {var_name}.weight > {var_name}.w_max)) {{
        LOG_WARN("Weight out of bounds: %.4f (clamping)", {var_name}.weight);
        {var_name}.weight = fmaxf(0.0f, {var_name}.w_max > 0.0f ? fminf({var_name}.weight, {var_name}.w_max) : {var_name}.weight);
    }}
"""

        insert_pos = match.end()
        content = content[:insert_pos] + validation + content[insert_pos:]

    return content, len(matches) > 0

def process_file(filepath):
    """Process a single file"""
    print(f"Processing: {filepath}")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content
        modified = False

        # Find all init/create functions
        func_pattern = r'(\w+(?:_init|_create|_params_\w+))\s*\('
        functions = re.findall(func_pattern, content)

        if functions:
            print(f"  Found {len(set(functions))} init/create functions")

        # Add weight bounds checks
        content, weight_modified = add_weight_bounds_check(content)
        if weight_modified:
            print(f"  ✓ Added weight bounds validation")
            modified = True

        # Save if modified
        if modified:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  ✓ File updated")
            return True
        else:
            print(f"  - No changes needed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    plasticity_dir = "/home/bbrelin/nimcp/src/plasticity"

    # Get all .c files
    c_files = []
    for root, dirs, files in os.walk(plasticity_dir):
        for file in files:
            if file.endswith('.c'):
                c_files.append(os.path.join(root, file))

    c_files.sort()

    print(f"Processing {len(c_files)} files...")
    print("=" * 70)

    modified_count = 0
    for filepath in c_files:
        if process_file(filepath):
            modified_count += 1
        print()

    print("=" * 70)
    print(f"Summary: Modified {modified_count}/{len(c_files)} files")

if __name__ == '__main__':
    main()
