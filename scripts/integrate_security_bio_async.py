#!/usr/bin/env python3
"""
Script to integrate bio-async into all Security module files
"""

import re
import sys

# Map of file names to their main struct names and module IDs
SECURITY_FILES = {
    'nimcp_cfi.c': {
        'struct': 'nimcp_cfi_context',
        'module': 'BIO_MODULE_CFI',
        'module_name': 'cfi',
        'create_func': 'nimcp_cfi_create',
        'destroy_func': 'nimcp_cfi_destroy'
    },
    'nimcp_continuous_monitor.c': {
        'struct': 'nimcp_continuous_monitor',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'continuous_monitor',
        'create_func': 'nimcp_continuous_monitor_create',
        'destroy_func': 'nimcp_continuous_monitor_destroy'
    },
    'nimcp_security_audit.c': {
        'struct': 'nimcp_audit_log',
        'module': 'BIO_MODULE_SECURITY_AUDIT',
        'module_name': 'security_audit',
        'create_func': 'nimcp_audit_log_create',
        'destroy_func': 'nimcp_audit_log_destroy'
    },
    'nimcp_security_coverage.c': {
        'struct': 'nimcp_security_coverage',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'security_coverage',
        'create_func': 'nimcp_security_coverage_create',
        'destroy_func': 'nimcp_security_coverage_destroy'
    },
    'nimcp_security_fractal.c': {
        'struct': 'nimcp_fractal_security',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'fractal_security',
        'create_func': 'nimcp_fractal_security_create',
        'destroy_func': 'nimcp_fractal_security_destroy'
    },
    'nimcp_security_integration.c': {
        'struct': 'nimcp_sec_integration',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'security_integration',
        'create_func': 'nimcp_sec_integration_create',
        'destroy_func': 'nimcp_sec_integration_destroy'
    },
    'nimcp_security_recovery_bridge.c': {
        'struct': 'nimcp_security_recovery_bridge',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'security_recovery_bridge',
        'create_func': 'nimcp_security_recovery_bridge_create',
        'destroy_func': 'nimcp_security_recovery_bridge_destroy'
    },
    'nimcp_shadow_stack.c': {
        'struct': 'nimcp_shadow_stack',
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'shadow_stack',
        'create_func': 'nimcp_shadow_stack_create',
        'destroy_func': 'nimcp_shadow_stack_destroy'
    },
    'nimcp_blood_brain_barrier.c': {
        'struct': None,  # Will find dynamically
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'blood_brain_barrier',
        'create_func': None,
        'destroy_func': None
    },
    'nimcp_bbb_access_control.c': {
        'struct': None,
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'bbb_access_control',
        'create_func': None,
        'destroy_func': None
    },
    'nimcp_bbb_code_signing.c': {
        'struct': None,
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'bbb_code_signing',
        'create_func': None,
        'destroy_func': None
    },
    'nimcp_bbb_input_gate.c': {
        'struct': None,
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'bbb_input_gate',
        'create_func': None,
        'destroy_func': None
    },
    'nimcp_bbb_memory_boundary.c': {
        'struct': None,
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'bbb_memory_boundary',
        'create_func': None,
        'destroy_func': None
    },
    'nimcp_security_math.c': {
        'struct': None,  # Multiple structs - will handle separately
        'module': 'BIO_MODULE_SECURITY',
        'module_name': 'security_math',
        'create_func': None,
        'destroy_func': None
    }
}

def add_bio_ctx_to_struct(content, struct_name):
    """Add bio_ctx fields to a struct"""
    # Find the struct definition
    pattern = rf'(struct {struct_name} {{[^}}]+?}})(;)'

    def replacer(match):
        struct_body = match.group(1)
        semicolon = match.group(2)

        # Check if already has bio_ctx
        if 'bio_ctx' in struct_body:
            return match.group(0)

        # Add bio_ctx fields before the closing brace
        bio_fields = '''
    // Bio-async integration
    bio_async_context_t* bio_ctx;
    bool bio_async_enabled;'''

        # Insert before the closing brace
        struct_body = struct_body.rstrip() + bio_fields + '\n'
        return struct_body + semicolon

    return re.sub(pattern, replacer, content, flags=re.DOTALL)

def add_registration_to_create(content, create_func, module_id, module_name):
    """Add bio-async registration to create function"""
    if not create_func:
        return content

    # Find the create function
    pattern = rf'({create_func}\([^{{]*\{{[^}}]+?return [^;]+;)(\s*}})'

    def replacer(match):
        func_body = match.group(1)
        closing = match.group(2)

        # Check if already has bio-async
        if 'bio_async_enabled' in func_body:
            return match.group(0)

        # Add bio-async registration before return
        registration = f'''
    // Bio-async integration
    result->bio_ctx = NULL;
    result->bio_async_enabled = false;
    bio_async_context_t* ctx = bio_router_get_global_context();
    if (ctx) {{
        result->bio_ctx = ctx;
        result->bio_async_enabled = bio_router_register_module(ctx, {module_id}, "{module_name}");
        LOG_INFO("Bio-async integration %s for {module_name}",
                 result->bio_async_enabled ? "enabled" : "failed");
    }}
'''

        # Find the last return statement
        lines = func_body.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if 'return' in lines[i]:
                lines.insert(i, registration)
                break

        func_body = '\n'.join(lines)
        return func_body + closing

    return re.sub(pattern, replacer, content, flags=re.DOTALL)

def add_unregistration_to_destroy(content, destroy_func, module_id):
    """Add bio-async unregistration to destroy function"""
    if not destroy_func:
        return content

    # Find the destroy function
    pattern = rf'({destroy_func}\([^{{]*\{{)([^}}]+)(}})'

    def replacer(match):
        func_start = match.group(1)
        func_body = match.group(2)
        func_end = match.group(3)

        # Check if already has unregistration
        if 'bio_router_unregister_module' in func_body:
            return match.group(0)

        # Add unregistration after initial null check
        unregistration = f'''
    // Bio-async unregistration
    if (instance->bio_async_enabled && instance->bio_ctx) {{
        bio_router_unregister_module(instance->bio_ctx, {module_id});
        LOG_INFO("Bio-async unregistered for {destroy_func.replace('_destroy', '')}");
    }}
'''

        # Insert after the first if (!instance) return;
        lines = func_body.split('\n')
        for i, line in enumerate(lines):
            if 'return' in line and ('!instance' in line or '!ctx' in line or 'instance == NULL' in line):
                lines.insert(i + 1, unregistration)
                break

        func_body = '\n'.join(lines)
        return func_start + func_body + func_end

    return re.sub(pattern, replacer, content, flags=re.DOTALL)

print("Security bio-async integration script")
print("This script adds bio_ctx fields and registration to security modules")
print("\nRun the Edit tool manually for each file based on the patterns above.")
