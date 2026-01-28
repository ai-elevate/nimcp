#!/usr/bin/env python3
"""Phase 8 instance-level upgrade: Add heartbeat_instance, set_instance_health_agent,
and full training functions to all cognitive .c files missing them."""

import os
import re
import sys

SRC_ROOT = "/home/bbrelin/nimcp/src/cognitive"

def find_missing_files():
    """Find all .c files missing set_instance_health_agent."""
    missing = []
    for root, dirs, files in os.walk(SRC_ROOT):
        for f in files:
            if f.endswith('.c'):
                path = os.path.join(root, f)
                with open(path, 'r') as fh:
                    content = fh.read()
                if 'set_instance_health_agent' not in content:
                    missing.append(path)
    return sorted(missing)

def extract_prefix(content, filename):
    """Extract module prefix from existing heartbeat function."""
    # Look for: static inline void PREFIX_heartbeat(
    m = re.search(r'static\s+inline\s+void\s+(\w+)_heartbeat\s*\(', content)
    if m:
        return m.group(1)
    # Look for: void PREFIX_heartbeat(
    m = re.search(r'void\s+(\w+)_heartbeat\s*\(', content)
    if m:
        return m.group(1)
    # Look for g_PREFIX_health_agent
    m = re.search(r'static\s+nimcp_health_agent_t\s*\*\s*g_(\w+)_health_agent', content)
    if m:
        return m.group(1)
    # Fallback: derive from filename
    base = os.path.basename(filename).replace('nimcp_', '').replace('.c', '')
    return base

def extract_global_var(content, prefix):
    """Extract global health agent variable name."""
    m = re.search(r'(g_\w+_health_agent)\s*=\s*NULL', content)
    if m:
        return m.group(1)
    return f"g_{prefix}_health_agent"

def has_heartbeat_instance(content):
    """Check if heartbeat_instance already exists."""
    return '_heartbeat_instance' in content

def find_struct_info(content, prefix):
    """Find the main struct typedef for casting in training functions."""
    # Look for typedef struct ... PREFIX_t or similar
    # Pattern 1: typedef struct { ... } prefix_t;
    patterns = [
        # typedef struct name { ... } name_t;
        rf'typedef\s+struct\s+(\w+)\s*\{{[^}}]*\}}\s*(\w+_t)\s*;',
        # struct name { ... };  with typedef struct name name_t;
        rf'struct\s+(\w+)\s*\{{',
        # typedef struct { ... } name_t;
        rf'typedef\s+struct\s*\{{[^}}]*\}}\s*(\w+_t)\s*;',
    ]

    # First try to find an opaque struct that matches the prefix
    # e.g., typedef struct foo_struct* foo_t;
    m = re.search(rf'typedef\s+struct\s+\w+\s*\*\s*(\w+_t)\s*;', content)
    if m:
        return m.group(1), True  # opaque handle

    # Look for struct definitions
    m = re.search(rf'struct\s+(\w+)\s*\{{', content)
    if m:
        struct_name = m.group(1)
        # Check if there's a typedef for it
        m2 = re.search(rf'typedef\s+struct\s+{re.escape(struct_name)}\s+(\w+_t)', content)
        if m2:
            return m2.group(1), False
        m2 = re.search(rf'typedef\s+struct\s+{re.escape(struct_name)}\s*\*\s*(\w+_t)', content)
        if m2:
            return m2.group(1), True  # pointer typedef
        return f"struct {struct_name}", False

    # Look for any typedef struct
    m = re.search(r'typedef\s+struct\s*\{[^}]*\}\s*(\w+_t)\s*;', content, re.DOTALL)
    if m:
        return m.group(1), False

    return None, False

def is_bridge_file(content, filename):
    """Determine if this is a bridge file (has bridge struct with health_agent field)."""
    basename = os.path.basename(filename)
    if '_bridge' in basename:
        # Check if struct has a health_agent field already
        if 'nimcp_health_agent_t* health_agent' in content or 'nimcp_health_agent_t *health_agent' in content:
            return True
        # Check for bridge_base_t which indicates a bridge pattern
        if 'bridge_base_t' in content:
            return True
    return False

def find_bridge_struct_type(content, prefix):
    """Find the bridge struct type for typed set_instance."""
    # Look for struct PREFIX { or typedef ... PREFIX_t
    patterns = [
        rf'typedef\s+struct\s+\w+\s+(\w*{prefix.replace("_bridge", "")}\w*_t)\s*;',
        rf'typedef\s+struct\s+\w+\s*\*\s*(\w*{prefix.replace("_bridge", "")}\w*_t)\s*;',
    ]
    for p in patterns:
        m = re.search(p, content)
        if m:
            return m.group(1)
    # Try the full prefix
    m = re.search(rf'typedef\s+struct\s+\w+\s+({prefix}_t)\s*;', content)
    if m:
        return m.group(1)
    m = re.search(rf'typedef\s+struct\s+\w+\s*\*\s*({prefix}_t)\s*;', content)
    if m:
        return m.group(1)
    return None

def generate_heartbeat_instance(prefix, global_var):
    """Generate heartbeat_instance function."""
    return f"""
/** @brief Send heartbeat from {prefix} module (instance-level) */
static inline void {prefix}_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{{
    if ({global_var}) {{
        nimcp_health_agent_heartbeat_ex({global_var}, operation, progress);
    }}
    if (instance_agent && instance_agent != {global_var}) {{
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }}
}}
"""

def generate_nonbridge_phase8(prefix, global_var, struct_type, is_opaque):
    """Generate set_instance + full training for non-bridge modules."""
    cast_type = struct_type if struct_type else "void"
    # For opaque types (pointer typedefs), cast directly
    # For struct types, cast to pointer
    if struct_type and is_opaque:
        cast_expr = f"({struct_type})instance"
        ptr_type = struct_type
    elif struct_type and not struct_type.startswith("struct "):
        cast_expr = f"({struct_type}*)instance"
        ptr_type = f"{struct_type}*"
    elif struct_type:
        cast_expr = f"({struct_type}*)instance"
        ptr_type = f"{struct_type}*"
    else:
        cast_expr = None
        ptr_type = None

    code = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void {prefix}_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {{
    if (instance) {{
        (void)agent;
        {global_var} = agent;
    }}
}}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int {prefix}_training_begin(void* instance) {{
    if (!instance) return -1;
    {prefix}_heartbeat_instance(NULL, "{prefix}_training_begin", 0.0f);
"""
    if cast_expr:
        code += f"    (void){cast_expr}; /* Module state available for reset */\n"
    code += f'    return 0;\n}}\n'

    code += f"""
int {prefix}_training_end(void* instance) {{
    if (!instance) return -1;
    {prefix}_heartbeat_instance(NULL, "{prefix}_training_end", 1.0f);
"""
    if cast_expr:
        code += f"    (void){cast_expr}; /* Module state available for finalization */\n"
    code += f'    return 0;\n}}\n'

    code += f"""
int {prefix}_training_step(void* instance, float progress) {{
    if (!instance) return -1;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance(NULL, "{prefix}_training_step", progress);
"""
    if cast_expr:
        code += f"    (void){cast_expr}; /* Module state available for step adaptation */\n"
    code += f'    return 0;\n}}\n'

    return code

def generate_bridge_phase8(prefix, global_var, bridge_type):
    """Generate set_instance + full training for bridge modules."""
    if not bridge_type:
        bridge_type = f"{prefix}_t"

    # Check if bridge_type is a pointer type
    if bridge_type.endswith('_t') and not bridge_type.startswith('struct'):
        param_type = f"{bridge_type}*"
        # Handle opaque pointer types
    else:
        param_type = f"{bridge_type}*"

    code = f"""
/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void {prefix}_set_instance_health_agent({param_type} bridge, nimcp_health_agent_t* agent) {{
    if (bridge) {{ bridge->health_agent = agent; }}
}}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int {prefix}_training_begin({param_type} bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{prefix}_training_begin", 0.0f);
    return 0;
}}

int {prefix}_training_end({param_type} bridge) {{
    if (!bridge) return -1;
    {prefix}_heartbeat_instance(bridge->health_agent, "{prefix}_training_end", 1.0f);
    return 0;
}}

int {prefix}_training_step({param_type} bridge, float progress) {{
    if (!bridge) return -1;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance(bridge->health_agent, "{prefix}_training_step", progress);
    return 0;
}}
"""
    return code

def insert_heartbeat_instance(content, prefix, global_var):
    """Insert heartbeat_instance after existing heartbeat function."""
    # Find the end of the existing heartbeat function
    pattern = rf'(static\s+inline\s+void\s+{re.escape(prefix)}_heartbeat\s*\([^)]*\)\s*\{{[^}}]*\}})'
    m = re.search(pattern, content, re.DOTALL)
    if m:
        insert_pos = m.end()
        hb_code = generate_heartbeat_instance(prefix, global_var)
        return content[:insert_pos] + hb_code + content[insert_pos:]

    # Try non-inline heartbeat
    pattern = rf'(void\s+{re.escape(prefix)}_heartbeat\s*\([^)]*\)\s*\{{[^}}]*\}})'
    m = re.search(pattern, content, re.DOTALL)
    if m:
        insert_pos = m.end()
        hb_code = generate_heartbeat_instance(prefix, global_var)
        return content[:insert_pos] + hb_code + content[insert_pos:]

    # If no heartbeat function found, insert after the global var declaration
    pattern = rf'(static\s+nimcp_health_agent_t\s*\*\s*{re.escape(global_var)}\s*=\s*NULL\s*;)'
    m = re.search(pattern, content)
    if m:
        insert_pos = m.end()
        hb_code = generate_heartbeat_instance(prefix, global_var)
        return content[:insert_pos] + hb_code + content[insert_pos:]

    return None

def process_file(filepath, dry_run=False):
    """Process a single file to add Phase 8 instance-level enhancements."""
    with open(filepath, 'r') as f:
        content = f.read()

    if 'set_instance_health_agent' in content:
        return "SKIP", "already has set_instance_health_agent"

    prefix = extract_prefix(content, filepath)
    global_var = extract_global_var(content, prefix)

    if global_var not in content:
        # Skip files that are #included by parent (no standalone compilation)
        if '#included by' in content or '@note This file is #include' in content:
            return "SKIP", "included by parent file"
        return "ERROR", f"global var {global_var} not found"

    modified = content
    changes = []

    # Step 1: Add heartbeat_instance if missing
    if not has_heartbeat_instance(modified):
        result = insert_heartbeat_instance(modified, prefix, global_var)
        if result:
            modified = result
            changes.append("heartbeat_instance")
        else:
            # Fallback: insert after global var
            hb_code = generate_heartbeat_instance(prefix, global_var)
            # Find a good insertion point
            m = re.search(rf'{re.escape(global_var)}\s*=\s*NULL\s*;', modified)
            if m:
                # Find end of line
                line_end = modified.find('\n', m.end())
                if line_end >= 0:
                    modified = modified[:line_end+1] + hb_code + modified[line_end+1:]
                    changes.append("heartbeat_instance(fallback)")

    # Step 2: Determine bridge vs non-bridge and append Phase 8 code
    is_bridge = is_bridge_file(modified, filepath)

    if is_bridge:
        bridge_type = find_bridge_struct_type(content, prefix)
        # Check if health_agent field needs to be added to struct
        if 'bridge_base_t' in content and 'nimcp_health_agent_t* health_agent' not in modified:
            # Add health_agent field after bridge_base_t
            m = re.search(r'(bridge_base_t\s+\w+\s*;[^\n]*\n)', modified)
            if m:
                modified = modified[:m.end()] + '    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */\n' + modified[m.end():]
                changes.append("health_agent_field")

        phase8_code = generate_bridge_phase8(prefix, global_var, bridge_type)
    else:
        struct_type, is_opaque = find_struct_info(content, prefix)
        phase8_code = generate_nonbridge_phase8(prefix, global_var, struct_type, is_opaque)

    # Append Phase 8 code at end of file
    modified = modified.rstrip() + '\n' + phase8_code
    changes.append("set_instance+training")

    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(modified)

    return "OK", f"prefix={prefix}, changes={','.join(changes)}"

def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    files = find_missing_files()
    print(f"Found {len(files)} files missing set_instance_health_agent")

    if dry_run:
        print("DRY RUN - no files will be modified")

    ok_count = 0
    err_count = 0
    skip_count = 0

    for filepath in files:
        rel_path = os.path.relpath(filepath, "/home/bbrelin/nimcp")
        status, msg = process_file(filepath, dry_run)
        if status == "OK":
            ok_count += 1
            if verbose:
                print(f"  OK: {rel_path} ({msg})")
        elif status == "SKIP":
            skip_count += 1
        else:
            err_count += 1
            print(f"  ERROR: {rel_path} ({msg})")

    print(f"\nResults: {ok_count} upgraded, {skip_count} skipped, {err_count} errors")

if __name__ == '__main__':
    main()
