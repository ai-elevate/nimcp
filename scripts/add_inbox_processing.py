#!/usr/bin/env python3
"""
Add bio_router_process_inbox calls to modules that register but don't process inbox.

This script adds inbox processing to the update/step/tick functions of modules.
"""

import re
import os
import sys

# Modules needing inbox processing (register but don't process)
MODULES_TO_FIX = [
    # Cognitive modules
    "src/cognitive/global_workspace/nimcp_global_workspace.c",
    "src/cognitive/executive/nimcp_executive.c",
    "src/cognitive/working_memory/nimcp_working_memory.c",
    "src/cognitive/emotions/nimcp_emotional_system.c",
    "src/cognitive/ethics/nimcp_ethics.c",
    "src/cognitive/curiosity/nimcp_curiosity.c",
    "src/cognitive/introspection/nimcp_introspection.c",
    "src/cognitive/consolidation/nimcp_consolidation.c",
    "src/cognitive/salience/nimcp_salience.c",
    "src/cognitive/knowledge/nimcp_knowledge.c",
    "src/cognitive/memory/nimcp_engram.c",
    "src/cognitive/memory/nimcp_semantic_memory.c",
    "src/cognitive/memory/nimcp_systems_consolidation.c",
    "src/cognitive/memory/nimcp_wm_transfer.c",
    "src/cognitive/predictive/nimcp_predictive.c",
    "src/cognitive/mirror_neurons/nimcp_mirror_neurons.c",
    "src/cognitive/theory_of_mind/nimcp_theory_of_mind.c",
    "src/cognitive/self_model/nimcp_self_model.c",
    # Middleware modules
    "src/middleware/events/nimcp_event_bus.c",
    "src/middleware/pipeline/nimcp_middleware_pipeline.c",
    "src/middleware/integration/nimcp_middleware_controller.c",
    # Glial modules
    "src/glial/astrocytes/nimcp_astrocytes.c",
    "src/glial/microglia/nimcp_microglia.c",
    "src/glial/oligodendrocytes/nimcp_oligodendrocytes.c",
]

# Pattern to find update/step/tick functions
UPDATE_FUNC_PATTERNS = [
    r'(_update|_step|_tick|_process)\s*\([^)]*\)\s*\{',
    r'(update|step|tick|process)_\w+\s*\([^)]*\)\s*\{',
]

# Pattern to find bio_ctx field access
BIO_CTX_PATTERNS = [
    r'(\w+)->bio_ctx',
    r'(\w+)->bio_async_ctx',
    r'(\w+)\.bio_ctx',
]

def find_bio_ctx_var(content):
    """Find the variable name that has bio_ctx."""
    for pattern in BIO_CTX_PATTERNS:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None

def find_update_function(content):
    """Find an update/step/tick function to add inbox processing to."""
    for pattern in UPDATE_FUNC_PATTERNS:
        match = re.search(pattern, content)
        if match:
            return match.start(), match.group(0)
    return None, None

def add_inbox_processing(filepath, base_path):
    """Add bio_router_process_inbox to a module's update function."""
    full_path = os.path.join(base_path, filepath)

    if not os.path.exists(full_path):
        print(f"SKIP: {filepath} - file not found")
        return False

    with open(full_path, 'r') as f:
        content = f.read()

    # Skip if already has inbox processing
    if 'bio_router_process_inbox' in content:
        print(f"SKIP: {filepath} - already has inbox processing")
        return False

    # Find the bio_ctx variable name
    ctx_var = find_bio_ctx_var(content)
    if not ctx_var:
        print(f"SKIP: {filepath} - no bio_ctx found")
        return False

    # Check if bio_async_enabled check exists
    has_bio_async_check = 'bio_async_enabled' in content

    # Find update function
    pos, func_match = find_update_function(content)
    if pos is None:
        # No update function - need to add one or find another approach
        print(f"WARN: {filepath} - no update function found, adding standalone process function")

        # Find a good insertion point (after register function or before unregister)
        insert_point = content.rfind('bio_router_unregister_module')
        if insert_point == -1:
            insert_point = len(content) - 1
        else:
            # Go back to start of the function containing unregister
            while insert_point > 0 and content[insert_point] != '\n':
                insert_point -= 1

        # Extract module name from filepath
        module_name = os.path.basename(filepath).replace('nimcp_', '').replace('.c', '')

        # Generate process_inbox function
        process_func = f'''
//=============================================================================
// Bio-Async Inbox Processing
//=============================================================================

/**
 * @brief Process pending bio-async messages for this module
 *
 * Call this function periodically from the module's update loop to process
 * incoming async messages.
 *
 * @param ctx Module context pointer
 * @param max_messages Maximum messages to process (0 = all)
 * @return Number of messages processed
 */
uint32_t {module_name}_process_inbox(void* ctx, uint32_t max_messages) {{
    if (!ctx) return 0;

    // Get bio_ctx from the module's internal structure
    // This assumes the struct has a bio_ctx or bio_async_ctx field
    bio_module_context_t bio_ctx = NULL;

    // Try to extract bio_ctx - this pattern varies by module
    // Most modules store it as a field in their context struct
    struct {{ void* _pad; bio_module_context_t bio_ctx; bool bio_async_enabled; }}* internal =
        (void*)((char*)ctx + sizeof(void*));  // Skip first pointer field

    if (internal && internal->bio_ctx && internal->bio_async_enabled) {{
        bio_ctx = internal->bio_ctx;
    }}

    if (!bio_ctx) return 0;

    return bio_router_process_inbox(bio_ctx, max_messages);
}}

'''
        # Insert before the unregister function
        new_content = content[:insert_point] + process_func + content[insert_point:]

        with open(full_path, 'w') as f:
            f.write(new_content)

        print(f"ADD: {filepath} - added standalone process_inbox function")
        return True

    # Found an update function - add inbox processing at the start
    # Find the opening brace
    brace_pos = content.find('{', pos)
    if brace_pos == -1:
        print(f"SKIP: {filepath} - couldn't find function body")
        return False

    # Insert inbox processing after the opening brace
    if has_bio_async_check:
        inbox_code = f'''
    // Process pending bio-async messages
    if ({ctx_var} && {ctx_var}->bio_async_enabled && {ctx_var}->bio_ctx) {{
        bio_router_process_inbox({ctx_var}->bio_ctx, 10);  // Process up to 10 messages per update
    }}
'''
    else:
        inbox_code = f'''
    // Process pending bio-async messages
    if ({ctx_var} && {ctx_var}->bio_ctx) {{
        bio_router_process_inbox({ctx_var}->bio_ctx, 10);  // Process up to 10 messages per update
    }}
'''

    new_content = content[:brace_pos+1] + inbox_code + content[brace_pos+1:]

    with open(full_path, 'w') as f:
        f.write(new_content)

    print(f"ADD: {filepath} - added inbox processing to update function")
    return True


def main():
    base_path = "/home/bbrelin/nimcp"

    if len(sys.argv) > 1:
        base_path = sys.argv[1]

    modified = 0
    skipped = 0

    for filepath in MODULES_TO_FIX:
        if add_inbox_processing(filepath, base_path):
            modified += 1
        else:
            skipped += 1

    print(f"\nSummary: Modified {modified}, Skipped {skipped}")
    return 0 if modified > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
