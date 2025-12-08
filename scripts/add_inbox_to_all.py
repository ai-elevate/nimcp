#!/usr/bin/env python3
"""
Add bio_router_process_inbox calls to all modules that register but don't process inbox.
"""

import os
import re
import sys

BASE_PATH = "/home/bbrelin/nimcp"

# Pattern to find update/step/tick/process functions
UPDATE_PATTERNS = [
    r'void\s+(\w+)_(update|step|tick|process)\s*\([^)]*\)\s*\{',
    r'void\s+(update|step|tick|process)_(\w+)\s*\([^)]*\)\s*\{',
    r'bool\s+(\w+)_(update|step|tick|process)\s*\([^)]*\)\s*\{',
    r'nimcp_error_t\s+(\w+)_(update|step|tick|process)\s*\([^)]*\)\s*\{',
]

# Pattern to find bio_ctx variable
BIO_CTX_PATTERNS = [
    (r'g_(\w+)_bio_ctx', 'global'),  # Global context like g_module_bio_ctx
    (r'g_(\w+)_bio_initialized', 'global_init'),  # Global initialized flag
    (r'(\w+)->bio_ctx', 'struct'),  # Struct member like ctx->bio_ctx
    (r'(\w+)->bio_async_ctx', 'struct'),  # Alternative struct member
    (r'(\w+)\.bio_ctx', 'struct_dot'),  # Dot notation
]

def find_bio_ctx_info(content):
    """Find bio_ctx variable and type."""
    for pattern, ctx_type in BIO_CTX_PATTERNS:
        match = re.search(pattern, content)
        if match:
            return match.group(0), ctx_type, match.group(1)
    return None, None, None

def find_init_check(content):
    """Find the bio_async_enabled or bio_initialized check pattern."""
    patterns = [
        r'(\w+)->bio_async_enabled',
        r'g_(\w+)_bio_initialized',
        r'bio_async_enabled',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(0)
    return None

def find_first_update_function(content):
    """Find the first update/step/tick/process function."""
    for pattern in UPDATE_PATTERNS:
        match = re.search(pattern, content)
        if match:
            # Find the opening brace position
            start = match.start()
            brace_pos = content.find('{', start)
            if brace_pos != -1:
                return brace_pos, match.group(0)
    return None, None

def add_inbox_processing(filepath):
    """Add bio_router_process_inbox to a module."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already has inbox processing
    if 'bio_router_process_inbox' in content:
        return False, "Already has inbox processing"

    # Skip if doesn't register
    if 'bio_router_register_module' not in content:
        return False, "Doesn't register with bio-router"

    # Find bio_ctx info
    bio_ctx_var, ctx_type, prefix = find_bio_ctx_info(content)
    if not bio_ctx_var:
        return False, "No bio_ctx found"

    # Find init check pattern
    init_check = find_init_check(content)

    # Find update function
    brace_pos, func_match = find_first_update_function(content)

    if brace_pos is None:
        # No update function - look for main processing functions
        # Try to find any function that processes data
        alt_patterns = [
            r'void\s+(\w+)_(compute|evaluate|apply|run|execute)\s*\([^)]*\)\s*\{',
            r'nimcp_error_t\s+(\w+)_(compute|evaluate|apply|run|execute)\s*\([^)]*\)\s*\{',
        ]
        for pattern in alt_patterns:
            match = re.search(pattern, content)
            if match:
                brace_pos = content.find('{', match.start())
                func_match = match.group(0)
                break

    if brace_pos is None:
        return False, "No suitable function found"

    # Build inbox processing code
    if ctx_type == 'global':
        if init_check and 'initialized' in init_check:
            inbox_code = f'''
    // Process pending bio-async messages
    if ({init_check} && {bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
'''
        else:
            inbox_code = f'''
    // Process pending bio-async messages
    if ({bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
'''
    elif ctx_type == 'struct':
        ctx_name = bio_ctx_var.split('->')[0]
        if init_check:
            inbox_code = f'''
    // Process pending bio-async messages
    if ({ctx_name} && {init_check} && {bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
'''
        else:
            inbox_code = f'''
    // Process pending bio-async messages
    if ({ctx_name} && {bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
'''
    else:
        inbox_code = f'''
    // Process pending bio-async messages
    if ({bio_ctx_var}) {{
        bio_router_process_inbox({bio_ctx_var}, 5);
    }}
'''

    # Insert after the opening brace
    new_content = content[:brace_pos+1] + inbox_code + content[brace_pos+1:]

    with open(filepath, 'w') as f:
        f.write(new_content)

    return True, f"Added to {func_match[:50]}..."

def main():
    # Get list of files that register but don't process inbox
    import subprocess

    # Files that register
    result = subprocess.run(
        ['grep', '-rl', 'bio_router_register_module', f'{BASE_PATH}/src', '--include=*.c'],
        capture_output=True, text=True
    )
    registered = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

    # Files that process inbox
    result = subprocess.run(
        ['grep', '-rl', 'bio_router_process_inbox', f'{BASE_PATH}/src', '--include=*.c'],
        capture_output=True, text=True
    )
    processing = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()

    # Files needing inbox processing
    to_fix = registered - processing

    print(f"Found {len(to_fix)} files needing inbox processing")
    print("=" * 60)

    modified = 0
    skipped = 0
    errors = []

    for filepath in sorted(to_fix):
        if not filepath:
            continue
        success, msg = add_inbox_processing(filepath)
        rel_path = filepath.replace(BASE_PATH + '/', '')
        if success:
            print(f"OK:   {rel_path}")
            modified += 1
        else:
            print(f"SKIP: {rel_path} - {msg}")
            skipped += 1
            if "No suitable function" in msg or "No bio_ctx" in msg:
                errors.append((rel_path, msg))

    print("=" * 60)
    print(f"Modified: {modified}, Skipped: {skipped}")

    if errors:
        print(f"\nFiles needing manual review ({len(errors)}):")
        for path, reason in errors:
            print(f"  {path}: {reason}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
