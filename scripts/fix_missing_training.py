#!/usr/bin/env python3
"""Fix files that have set_instance_health_agent but are missing training functions."""
import os
import re

SRC_ROOT = "/home/bbrelin/nimcp/src/cognitive"

def find_files_missing_training():
    """Find files with set_instance but missing training functions."""
    results = {'begin': [], 'end': [], 'step': []}
    for root, dirs, files in os.walk(SRC_ROOT):
        for f in files:
            if not f.endswith('.c'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            if 'set_instance_health_agent' not in content:
                continue
            if 'training_begin' not in content:
                results['begin'].append(path)
            if 'training_end' not in content:
                results['end'].append(path)
            if 'training_step' not in content:
                results['step'].append(path)
    return results

def extract_prefix(content):
    m = re.search(r'void\s+(\w+)_set_instance_health_agent\s*\(', content)
    if m:
        return m.group(1)
    m = re.search(r'static\s+inline\s+void\s+(\w+)_heartbeat_instance\s*\(', content)
    if m:
        return m.group(1)
    return None

def is_bridge(content):
    return 'bridge_base_t' in content or ('_bridge' in content and 'health_agent' in content)

def get_bridge_type(content, prefix):
    # Look for the set_instance signature to determine type
    m = re.search(rf'void\s+{re.escape(prefix)}_set_instance_health_agent\s*\(\s*(\w+(?:\s*\*)?)\s+\w+', content)
    if m:
        return m.group(1).strip()
    return 'void*'

def fix_file(path):
    with open(path) as f:
        content = f.read()

    prefix = extract_prefix(content)
    if not prefix:
        print(f"  SKIP: {path} (no prefix found)")
        return

    has_begin = 'training_begin' in content
    has_end = 'training_end' in content
    has_step = 'training_step' in content

    bridge = is_bridge(content)
    param_type = get_bridge_type(content, prefix) if bridge else 'void*'
    param_name = 'bridge' if bridge else 'instance'

    # Determine heartbeat call pattern
    if bridge and param_type != 'void*':
        hb_agent = f"{param_name}->health_agent"
    else:
        hb_agent = "NULL"

    additions = []

    if not has_begin:
        additions.append(f"""
int {prefix}_training_begin({param_type} {param_name}) {{
    if (!{param_name}) return -1;
    {prefix}_heartbeat_instance({hb_agent}, "{prefix}_training_begin", 0.0f);
    return 0;
}}
""")

    if not has_end:
        additions.append(f"""
int {prefix}_training_end({param_type} {param_name}) {{
    if (!{param_name}) return -1;
    {prefix}_heartbeat_instance({hb_agent}, "{prefix}_training_end", 1.0f);
    return 0;
}}
""")

    if not has_step:
        additions.append(f"""
int {prefix}_training_step({param_type} {param_name}, float progress) {{
    if (!{param_name}) return -1;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    {prefix}_heartbeat_instance({hb_agent}, "{prefix}_training_step", progress);
    return 0;
}}
""")

    if additions:
        content = content.rstrip() + '\n' + ''.join(additions)
        with open(path, 'w') as f:
            f.write(content)
        missing = []
        if not has_begin: missing.append('begin')
        if not has_end: missing.append('end')
        if not has_step: missing.append('step')
        print(f"  FIXED: {os.path.relpath(path, '/home/bbrelin/nimcp')} (added training_{','.join(missing)})")

def main():
    results = find_files_missing_training()
    all_files = set(results['begin'] + results['end'] + results['step'])
    print(f"Found {len(all_files)} files needing training function fixes")
    for path in sorted(all_files):
        fix_file(path)

if __name__ == '__main__':
    main()
