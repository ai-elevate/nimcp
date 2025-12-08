#!/usr/bin/env python3
"""
Add bio-async registration to cognitive modules.

This script adds bio_module_context_t fields to structs and registration/unregistration
code to create/destroy functions in cognitive modules.
"""

import re
import sys
import os

# Mapping of module file patterns to their BIO_MODULE_* enum names
MODULE_MAPPINGS = {
    'autobiographical_memory': 'BIO_MODULE_MEMORY',
    'bias_detection': 'BIO_MODULE_EMOTIONS',
    'empathetic_response': 'BIO_MODULE_EMPATHETIC_RESPONSE',
    'explanations': 'BIO_MODULE_KNOWLEDGE',
    'emotional_tagging': 'BIO_MODULE_EMOTIONAL_TAGGING',
    'failure_prediction': 'BIO_MODULE_SYSTEM',
    'fault_attention': 'BIO_MODULE_ATTENTION',
    'fault_working_memory': 'BIO_MODULE_WORKING_MEMORY',
    'metacognition': 'BIO_MODULE_INTROSPECTION',
    'recovery_consolidation': 'BIO_MODULE_CONSOLIDATION',
    'recovery_episodic_memory': 'BIO_MODULE_MEMORY',
    'recovery_executive': 'BIO_MODULE_EXECUTIVE',
    'grief_and_loss': 'BIO_MODULE_GRIEF',
    'joy_euphoria': 'BIO_MODULE_JOY',
    'symbolic_logic': 'BIO_MODULE_KNOWLEDGE',
    'engram': 'BIO_MODULE_MEMORY',
    'semantic_memory': 'BIO_MODULE_MEMORY',
    'systems_consolidation': 'BIO_MODULE_CONSOLIDATION',
    'wm_transfer': 'BIO_MODULE_WORKING_MEMORY',
    'disorder_detectors': 'BIO_MODULE_WELLBEING',
    'interventions': 'BIO_MODULE_WELLBEING',
    'mental_health': 'BIO_MODULE_WELLBEING',
    'meta_learning': 'BIO_MODULE_KNOWLEDGE',
    'personality': 'BIO_MODULE_EMOTIONS',
    'reasoning_attention': 'BIO_MODULE_ATTENTION',
    'reasoning_curiosity': 'BIO_MODULE_CURIOSITY',
    'backward_chaining': 'BIO_MODULE_KNOWLEDGE',
    'forward_chaining': 'BIO_MODULE_KNOWLEDGE',
    'knowledge_base_interface': 'BIO_MODULE_KNOWLEDGE',
    'reasoning_factory': 'BIO_MODULE_KNOWLEDGE',
    'reasoning_integration': 'BIO_MODULE_KNOWLEDGE',
    'symbolic_logic_attachment': 'BIO_MODULE_KNOWLEDGE',
    'symbolic_logic_brain_integration': 'BIO_MODULE_KNOWLEDGE',
    'unification_engine': 'BIO_MODULE_KNOWLEDGE',
    'remorse_regret': 'BIO_MODULE_REMORSE',
    'self_awareness_extended': 'BIO_MODULE_INTROSPECTION',
    'self_model': 'BIO_MODULE_INTROSPECTION',
    'shadow_emotions': 'BIO_MODULE_EMOTIONS',
    'sleep_wake': 'BIO_MODULE_CONSOLIDATION',
    'love_loyalty_friendship': 'BIO_MODULE_EMOTIONS',
    'theory_of_mind': 'BIO_MODULE_INTROSPECTION',
}

def get_module_name(filepath):
    """Extract module name from file path."""
    basename = os.path.basename(filepath)
    # Remove nimcp_ prefix and .c suffix
    name = basename.replace('nimcp_', '').replace('.c', '')
    return name

def get_bio_module_id(filepath):
    """Get the BIO_MODULE_* enum name for a file."""
    name = get_module_name(filepath)
    return MODULE_MAPPINGS.get(name, 'BIO_MODULE_SYSTEM')

def add_struct_fields(content, module_name):
    """Add bio_ctx and bio_async_enabled to struct if not present."""
    if 'bio_module_context_t' in content and 'bio_async_enabled' in content:
        return content, False

    # Find struct patterns (looking for }; after struct definition)
    # Pattern: struct something { ... };
    struct_pattern = r'(struct\s+\w+\s*\{[^}]+)(};)'

    def add_bio_fields(match):
        struct_content = match.group(1)
        struct_end = match.group(2)

        # Check if already has bio fields
        if 'bio_module_context_t' in struct_content:
            return match.group(0)

        # Add bio-async fields before closing
        bio_fields = """
    // Bio-async integration
    bio_module_context_t bio_ctx;   /**< Bio-async module context */
    bool bio_async_enabled;         /**< Bio-async registration status */
"""
        return struct_content + bio_fields + struct_end

    new_content, count = re.subn(struct_pattern, add_bio_fields, content, flags=re.DOTALL)
    return new_content, count > 0

def find_create_function(content, module_name):
    """Find the create/init function for a module."""
    # Common patterns for create functions
    patterns = [
        rf'\w+_create_custom\s*\([^)]*\)\s*\{{',
        rf'\w+_create\s*\([^)]*\)\s*\{{',
        rf'\w+_init\s*\([^)]*\)\s*\{{',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.start()
    return None

def add_registration_to_create(content, module_name, bio_module_id):
    """Add bio-router registration to create function."""
    if 'bio_router_register_module' in content:
        return content, False

    # Find return statements in create functions that return the instance
    # Look for patterns like "return wm;" or "return exec;"
    create_funcs = ['_create_custom', '_create', '_init']

    for func_suffix in create_funcs:
        # Find function boundaries
        func_pattern = rf'(\w+{func_suffix})\s*\([^)]*\)\s*\{{'
        match = re.search(func_pattern, content)
        if not match:
            continue

        func_start = match.start()
        func_name = match.group(1)

        # Find the matching closing brace
        brace_count = 0
        func_end = -1
        in_func = False
        for i, char in enumerate(content[func_start:], func_start):
            if char == '{':
                brace_count += 1
                in_func = True
            elif char == '}':
                brace_count -= 1
                if in_func and brace_count == 0:
                    func_end = i
                    break

        if func_end == -1:
            continue

        func_content = content[func_start:func_end+1]

        # Find the last "return X;" where X is a variable (not NULL)
        return_matches = list(re.finditer(r'return\s+(\w+)\s*;', func_content))

        # Filter out "return NULL;" matches
        valid_returns = [m for m in return_matches if m.group(1) != 'NULL']

        if not valid_returns:
            continue

        last_return = valid_returns[-1]
        var_name = last_return.group(1)

        # Get short module name for logging
        short_name = module_name.replace('nimcp_', '')

        # Create registration code
        registration_code = f'''
    // Bio-async registration
    {var_name}->bio_ctx = NULL;
    {var_name}->bio_async_enabled = false;
    if (bio_router_is_initialized()) {{
        bio_module_info_t bio_info = {{
            .module_id = {bio_module_id},
            .module_name = "{short_name}",
            .inbox_capacity = 32,
            .user_data = {var_name}
        }};
        {var_name}->bio_ctx = bio_router_register_module(&bio_info);
        if ({var_name}->bio_ctx) {{
            {var_name}->bio_async_enabled = true;
        }}
    }}

'''
        # Insert before the return statement
        insert_pos = func_start + last_return.start()
        new_content = content[:insert_pos] + registration_code + content[insert_pos:]
        return new_content, True

    return content, False

def add_unregister_to_destroy(content, module_name):
    """Add bio-router unregistration to destroy function."""
    if 'bio_router_unregister_module' in content:
        return content, False

    # Find destroy functions
    destroy_patterns = [
        rf'(\w+_destroy)\s*\(\s*\w+\s*\*\s*(\w+)\s*\)\s*\{{',
        rf'(\w+_free)\s*\(\s*\w+\s*\*\s*(\w+)\s*\)\s*\{{',
        rf'(\w+_cleanup)\s*\(\s*\w+\s*\*\s*(\w+)\s*\)\s*\{{',
    ]

    for pattern in destroy_patterns:
        match = re.search(pattern, content)
        if not match:
            continue

        func_name = match.group(1)
        param_name = match.group(2)
        func_start = match.start()

        # Find the last nimcp_free(param_name) call
        # This is typically before freeing the main structure
        free_pattern = rf'nimcp_free\s*\(\s*{param_name}\s*\)\s*;'
        free_matches = list(re.finditer(free_pattern, content[func_start:]))

        if not free_matches:
            continue

        # Insert unregister code before the last free
        last_free = free_matches[-1]
        insert_pos = func_start + last_free.start()

        unregister_code = f'''// Unregister from bio-router
    if ({param_name}->bio_async_enabled && {param_name}->bio_ctx) {{
        bio_router_unregister_module({param_name}->bio_ctx);
        {param_name}->bio_ctx = NULL;
        {param_name}->bio_async_enabled = false;
    }}

    '''

        new_content = content[:insert_pos] + unregister_code + content[insert_pos:]
        return new_content, True

    return content, False

def process_file(filepath):
    """Process a single file to add bio-async integration."""
    print(f"Processing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    module_name = get_module_name(filepath)
    bio_module_id = get_bio_module_id(filepath)

    modified = False

    # Add struct fields
    content, changed = add_struct_fields(content, module_name)
    modified = modified or changed

    # Add registration to create function
    content, changed = add_registration_to_create(content, module_name, bio_module_id)
    modified = modified or changed

    # Add unregister to destroy function
    content, changed = add_unregister_to_destroy(content, module_name)
    modified = modified or changed

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Modified: {filepath}")
    else:
        print(f"  Skipped (already integrated or no matching patterns): {filepath}")

    return modified

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python add_bio_async_cognitive.py <file1.c> [file2.c ...]")
        print("Or: python add_bio_async_cognitive.py --all")
        sys.exit(1)

    if sys.argv[1] == '--all':
        # Find all cognitive files that need integration
        import subprocess
        result = subprocess.run(
            ['find', 'src/cognitive', '-name', '*.c', '-exec', 'grep', '-l', 'bio_async.h', '{}', ';'],
            capture_output=True, text=True
        )
        files = result.stdout.strip().split('\n')

        # Filter out files that already have registration
        files_to_process = []
        for f in files:
            if not f:
                continue
            with open(f, 'r') as fp:
                if 'bio_router_register_module' not in fp.read():
                    files_to_process.append(f)

        print(f"Found {len(files_to_process)} files to process")
    else:
        files_to_process = sys.argv[1:]

    modified_count = 0
    for filepath in files_to_process:
        if process_file(filepath):
            modified_count += 1

    print(f"\nModified {modified_count}/{len(files_to_process)} files")

if __name__ == '__main__':
    main()
