#!/usr/bin/env python3
"""
Add full bio-async messaging to cognitive modules.

This script adds:
1. Message handlers for relevant message types
2. Send/broadcast calls at key events
3. Handler function implementations
"""

import os
import re

# Module configurations: module_name -> (handler_types, send_events, broadcast_events)
COGNITIVE_MODULES = {
    'working_memory': {
        'handlers': [
            ('BIO_MSG_WORKING_MEMORY_STORE', 'handle_wm_store_request'),
            ('BIO_MSG_WORKING_MEMORY_RETRIEVE', 'handle_wm_retrieve_request'),
        ],
        'broadcasts': [
            ('on_item_added', 'BIO_MSG_WORKING_MEMORY_STORE', 'item stored in working memory'),
            ('on_item_evicted', 'BIO_MSG_ATTENTION_SHIFT', 'item evicted from working memory'),
        ],
    },
    'introspection': {
        'handlers': [
            ('BIO_MSG_INTROSPECTION_QUERY', 'handle_introspection_query'),
        ],
        'broadcasts': [
            ('on_state_change', 'BIO_MSG_INTROSPECTION_RESPONSE', 'introspection state changed'),
        ],
    },
    'salience': {
        'handlers': [
            ('BIO_MSG_SALIENCE_QUERY', 'handle_salience_query'),
        ],
        'broadcasts': [
            ('on_high_salience', 'BIO_MSG_SALIENCE_RESPONSE', 'high salience detected'),
        ],
    },
    'ethics': {
        'handlers': [
            ('BIO_MSG_ETHICS_EVALUATION_REQUEST', 'handle_ethics_request'),
        ],
        'broadcasts': [
            ('on_veto', 'BIO_MSG_ETHICS_EVALUATION_RESPONSE', 'ethics veto issued'),
        ],
    },
    'knowledge': {
        'handlers': [
            ('BIO_MSG_KNOWLEDGE_QUERY', 'handle_knowledge_query'),
        ],
        'broadcasts': [
            ('on_knowledge_update', 'BIO_MSG_KNOWLEDGE_RESPONSE', 'knowledge base updated'),
        ],
    },
    'curiosity': {
        'handlers': [],
        'broadcasts': [
            ('on_curiosity_spike', 'BIO_MSG_CURIOSITY_SIGNAL', 'curiosity signal generated'),
        ],
    },
    'executive': {
        'handlers': [
            ('BIO_MSG_DECISION_REQUEST', 'handle_decision_request'),
        ],
        'broadcasts': [
            ('on_decision', 'BIO_MSG_DECISION_RESPONSE', 'executive decision made'),
        ],
    },
    'consolidation': {
        'handlers': [
            ('BIO_MSG_CONSOLIDATION_TRIGGER', 'handle_consolidation_trigger'),
        ],
        'broadcasts': [],
    },
    'mirror_neurons': {
        'handlers': [
            ('BIO_MSG_MIRROR_NEURON_ACTIVATION', 'handle_mirror_activation'),
        ],
        'broadcasts': [
            ('on_mirror_fire', 'BIO_MSG_MIRROR_NEURON_ACTIVATION', 'mirror neuron fired'),
        ],
    },
    'global_workspace': {
        'handlers': [
            ('BIO_MSG_ATTENTION_SHIFT', 'handle_attention_shift'),
        ],
        'broadcasts': [
            ('on_broadcast', 'BIO_MSG_ATTENTION_SHIFT', 'global workspace broadcast'),
        ],
    },
}

# Handler template
HANDLER_TEMPLATE = '''
/**
 * @brief Bio-async message handler: {desc}
 */
static nimcp_error_t {handler_name}(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{{
    (void)msg_size;
    (void)response_promise;

    if (!msg || !user_data) {{
        return NIMCP_ERROR_INVALID_ARG;
    }}

    // TODO: Implement {handler_name} logic
    LOG_DEBUG("Received {msg_type} message");

    return NIMCP_SUCCESS;
}}
'''

# Broadcast helper template
BROADCAST_HELPER = '''
/**
 * @brief Broadcast {desc} event via bio-async
 */
static void bio_broadcast_{event_name}({struct_type}* ctx) {{
    if (!ctx || !ctx->bio_async_enabled || !ctx->bio_ctx) {{
        return;
    }}

    {msg_type_struct} msg = {{}};
    bio_msg_init_header(&msg.header, {msg_type},
                        bio_module_context_get_id(ctx->bio_ctx),
                        0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;

    bio_router_broadcast(ctx->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG("Broadcast {msg_type} event");
}}
'''

# Registration code to add after bio_router_register_module
HANDLER_REGISTRATION = '''
        // Register message handlers
{registrations}
'''

def get_struct_type(module_name):
    """Get the main struct type for a module."""
    type_map = {
        'working_memory': 'working_memory_t',
        'introspection': 'introspection_context_t',
        'salience': 'salience_system_t',
        'ethics': 'ethics_system_t',
        'knowledge': 'knowledge_system_t',
        'curiosity': 'curiosity_engine_t',
        'executive': 'executive_controller_t',
        'consolidation': 'consolidation_system_t',
        'mirror_neurons': 'mirror_neurons_t',
        'global_workspace': 'global_workspace_t',
    }
    return type_map.get(module_name, 'void')

def get_msg_struct_type(msg_type):
    """Map message type enum to struct type."""
    struct_map = {
        'BIO_MSG_WORKING_MEMORY_STORE': 'bio_msg_wm_store_t',
        'BIO_MSG_WORKING_MEMORY_RETRIEVE': 'bio_msg_wm_retrieve_t',
        'BIO_MSG_INTROSPECTION_QUERY': 'bio_msg_introspection_query_t',
        'BIO_MSG_INTROSPECTION_RESPONSE': 'bio_msg_introspection_response_t',
        'BIO_MSG_SALIENCE_QUERY': 'bio_msg_salience_query_t',
        'BIO_MSG_SALIENCE_RESPONSE': 'bio_msg_salience_response_t',
        'BIO_MSG_ETHICS_EVALUATION_REQUEST': 'bio_msg_ethics_request_t',
        'BIO_MSG_ETHICS_EVALUATION_RESPONSE': 'bio_msg_ethics_response_t',
        'BIO_MSG_KNOWLEDGE_QUERY': 'bio_msg_introspection_query_t',  # Reuse
        'BIO_MSG_KNOWLEDGE_RESPONSE': 'bio_msg_introspection_response_t',
        'BIO_MSG_CURIOSITY_SIGNAL': 'bio_msg_curiosity_signal_t',
        'BIO_MSG_DECISION_REQUEST': 'bio_msg_introspection_query_t',
        'BIO_MSG_DECISION_RESPONSE': 'bio_msg_introspection_response_t',
        'BIO_MSG_CONSOLIDATION_TRIGGER': 'bio_msg_introspection_query_t',
        'BIO_MSG_MIRROR_NEURON_ACTIVATION': 'bio_msg_introspection_query_t',
        'BIO_MSG_ATTENTION_SHIFT': 'bio_msg_attention_shift_t',
    }
    return struct_map.get(msg_type, 'bio_message_header_t')

def find_module_file(module_name):
    """Find the .c file for a module."""
    base_path = '/home/bbrelin/nimcp/src/cognitive'

    # Direct mapping
    direct_paths = {
        'working_memory': 'working_memory/nimcp_working_memory.c',
        'introspection': 'introspection/nimcp_introspection.c',
        'salience': 'salience/nimcp_salience.c',
        'ethics': 'ethics/nimcp_ethics.c',
        'knowledge': 'knowledge/nimcp_knowledge.c',
        'curiosity': 'curiosity/nimcp_curiosity.c',
        'executive': 'executive/nimcp_executive.c',
        'consolidation': 'consolidation/nimcp_consolidation.c',
        'mirror_neurons': 'mirror_neurons/nimcp_mirror_neurons.c',
        'global_workspace': 'global_workspace/nimcp_global_workspace.c',
    }

    if module_name in direct_paths:
        return os.path.join(base_path, direct_paths[module_name])
    return None

def add_async_to_module(module_name, config):
    """Add full async messaging to a module."""
    filepath = find_module_file(module_name)
    if not filepath or not os.path.exists(filepath):
        print(f"  Skipping {module_name}: file not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already has full async
    if 'bio_router_register_handler' in content:
        print(f"  {module_name}: already has handlers")
        return False

    # Check if has registration
    if 'bio_router_register_module' not in content:
        print(f"  {module_name}: no registration, skipping")
        return False

    struct_type = get_struct_type(module_name)
    modified = False

    # 1. Add handler function declarations before create function
    handlers_code = []
    for msg_type, handler_name in config.get('handlers', []):
        handler = HANDLER_TEMPLATE.format(
            desc=f"Handle {msg_type}",
            handler_name=handler_name,
            msg_type=msg_type
        )
        handlers_code.append(handler)

    # 2. Add broadcast helper functions
    for event_name, msg_type, desc in config.get('broadcasts', []):
        broadcast = BROADCAST_HELPER.format(
            desc=desc,
            event_name=event_name,
            struct_type=struct_type,
            msg_type=msg_type,
            msg_type_struct=get_msg_struct_type(msg_type)
        )
        handlers_code.append(broadcast)

    # Find a good insertion point (before the create function)
    create_pattern = re.compile(
        rf'(?=\n[^\n]*{module_name.replace("_", ".*")}.*create[^\n]*\([^)]*\)\s*\{{)',
        re.IGNORECASE
    )

    # Alternative: insert after includes but before first function
    if handlers_code:
        # Find insertion point after includes
        include_end = content.rfind('#include')
        if include_end != -1:
            include_end = content.find('\n', include_end) + 1
            # Find next blank line after includes
            next_section = content.find('\n\n', include_end)
            if next_section != -1:
                handlers_section = '\n//=============================================================================\n'
                handlers_section += '// BIO-ASYNC MESSAGE HANDLERS\n'
                handlers_section += '//=============================================================================\n'
                handlers_section += ''.join(handlers_code)

                content = content[:next_section] + handlers_section + content[next_section:]
                modified = True

    # 3. Add handler registrations after bio_router_register_module success check
    if config.get('handlers'):
        reg_code = ''
        for msg_type, handler_name in config['handlers']:
            reg_code += f'        bio_router_register_handler(ctx->bio_ctx, {msg_type}, {handler_name});\n'

        # Find the success check after registration
        pattern = r"(if\s*\([^)]*bio_ctx[^)]*\)\s*\{[^}]*bio_async_enabled\s*=\s*true;[^}]*\})"

        def add_registrations(match):
            block = match.group(1)
            # Insert before the closing brace
            insert_pos = block.rfind('}')
            return block[:insert_pos] + '\n' + reg_code + '        ' + block[insert_pos:]

        new_content = re.sub(pattern, add_registrations, content, count=1)
        if new_content != content:
            content = new_content
            modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Modified: {module_name}")
        return True

    print(f"  {module_name}: no changes needed")
    return False

def main():
    os.chdir('/home/bbrelin/nimcp')

    print("Adding full bio-async messaging to cognitive modules...")
    modified = 0

    for module_name, config in COGNITIVE_MODULES.items():
        print(f"Processing {module_name}...")
        if add_async_to_module(module_name, config):
            modified += 1

    print(f"\nModified {modified}/{len(COGNITIVE_MODULES)} modules")

if __name__ == '__main__':
    main()
